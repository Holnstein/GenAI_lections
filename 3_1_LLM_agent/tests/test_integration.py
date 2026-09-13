# tests/test_integration.py

import requests
import psycopg2
import pytest

from decouple import config

from llm_agent.tool_sql_query import SQLQueryTool


def _postgres_available() -> bool:
    """Проверяет доступность PostgreSQL с параметрами из .env."""
    try:
        conn = psycopg2.connect(
            host=config("PG_HOST", default="localhost"),
            port=config("PG_PORT", default="5432"),
            dbname=config("PG_DATABASE", default="postgres"),
            user=config("PG_USER", default="postgres"),
            password=config("PG_PASSWORD", default="postgres"),
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


def _ollama_available() -> bool:
    """Проверяет доступность Ollama."""
    base_url = config("OLLAMA_BASE_URL", default="http://localhost:11434")
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


RUN_INTEGRATION = str(
    config("RUN_INTEGRATION", default="0")
).lower() in ("1", "true", "yes")

integration = pytest.mark.skipif(
    not (RUN_INTEGRATION and _postgres_available()),
    reason="Интеграционные тесты выключены: нужен RUN_INTEGRATION=1, PostgreSQL и Ollama",
)


class TestIntegration:
    """Интеграционные тесты с реальной БД и сервером Ollama."""

    @integration
    def test_db_schema_and_query(self):
        """Реальная БД: создание таблицы customers с данными и выполнение SQL."""
        with SQLQueryTool(local=True, auto_create_schema=True) as tool:
            rows = tool._execute_sql(
                "SELECT first_name, email FROM customers ORDER BY id"
            )
        assert len(rows) == 5
        assert rows[0]["first_name"] == "Иван"
        assert rows[0]["email"] == "ivan@mail.ru"

    @integration
    def test_ollama_is_up(self):
        """Реальный Ollama: сервер отвечает, клиент определяет его как доступный."""
        with SQLQueryTool(local=True) as tool:
            assert tool.test_ollama_connection() is True

    @integration
    def test_end_to_end_use_with_ollama(self):
        """Полный цикл: вопрос -> LLM генерирует SQL -> результат из БД."""
        with SQLQueryTool(local=True, auto_create_schema=True, timeout=300) as tool:
            result = None
            for _ in range(3):
                result = tool.use("Покажи имена и email всех клиентов")
                if "ivan@mail.ru" in result:
                    break
        assert "ivan@mail.ru" in result