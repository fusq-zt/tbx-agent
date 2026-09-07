from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event, current_thread

from tbx_agent.config import Settings
from tbx_agent.orchestration import TBXAgentGraph
from tbx_agent.service import TBXAgentService


def test_graph_finalization_cannot_erase_a_concurrent_screening_session(tmp_path, monkeypatch):
    settings = replace(
        Settings.from_env(),
        data_root=tmp_path,
        db_path=tmp_path / "state.sqlite3",
        artifact_root=tmp_path / "artifacts",
        vision_backend="mock",
        narrator_backend="none",
        require_real_inference=False,
        require_llm_inference=False,
        anatomy_backend="none",
        anatomy_required=False,
    )
    service = TBXAgentService(settings)
    before_final_save, release_final_save, screening_saved = Event(), Event(), Event()
    original_save = service.store.save_thread

    def save_thread(state):
        if current_thread().name.startswith("graph-turn") and state.recent_messages:
            before_final_save.set()
            assert release_final_save.wait(timeout=5)
        result = original_save(state)
        if state.active_screening_session_id is not None:
            screening_saved.set()
        return result

    monkeypatch.setattr(service.store, "save_thread", save_thread)
    identity = {"thread_id": "thread-1", "user_id": "user", "owner_scope": "tenant:user"}
    try:
        with (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="graph-turn") as graph_worker,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="screening") as screening_worker,
        ):
            graph = graph_worker.submit(
                TBXAgentGraph(service).invoke_with_receipt,
                {**identity, "message": "你能做什么？"},
            )
            try:
                assert before_final_save.wait(timeout=3)
                screening = screening_worker.submit(
                    service.start_active_screening, **identity, consent=True
                )
                # With different lock domains the new session is committed
                # before the graph writes its detached, older ThreadState.
                screening_saved.wait(timeout=0.5)
            finally:
                release_final_save.set()
            graph.result(timeout=5)
            session, _ = screening.result(timeout=5)
        thread = service.store.get_or_create_thread(**identity)
        assert thread.active_screening_session_id == session.session_id
        assert thread.recent_messages
    finally:
        release_final_save.set()
        service.tool_registry.close()
        service.store.close()
