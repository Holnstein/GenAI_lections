# tests/test_sql_query_tool.py

import sys
import os
from pathlib import Path

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

# Импортируем классы
from llm_agent.tool_sql_query import SQLQueryTool, SQLGenerationError, UnsafeSQLError


class TestSQLQueryTool:
    """Тесты для SQLQueryTool."""
    
    @pytest.fixture
    def mock_tool(self):
        """Создает экземпляр SQLQueryTool с моками для БД."""
        with patch('psycopg2.connect') as mock_connect:
            # Создаем мок для курсора
            mock_cursor = MagicMock()
            mock_connection = MagicMock()
            mock_connection.cursor.return_value.__enter__.return_value = mock_cursor
            mock_connect.return_value = mock_connection
            
            tool = SQLQueryTool(
                local=True,
                ollama_model="qwen3-vl:2b",
                pg_host="localhost",
                pg_port="5432",
                pg_db="testdb",
                pg_user="postgres",
                pg_password="postgres",
                auto_create_schema=False,  # Не создаем схему автоматически
            )
            
            # Подменяем методы для тестов
            tool._make_api_request = MagicMock()
            
            yield tool
            tool.close()
    
    @pytest.fixture
    def sql_tool(self):
        """Создает изолированный экземпляр с моком БД для юнит-тестов."""
        with patch('psycopg2.connect'):
            tool = SQLQueryTool(
                local=True,
                ollama_model="qwen3-vl:2b",
                pg_host="localhost",
                pg_port="5432",
                pg_db="testdb",
                pg_user="postgres",
                pg_password="postgres",
                auto_create_schema=False,
            )
        yield tool
        tool.close()
    
    def test_name_and_description(self, sql_tool):
        """Тест 1: Проверка имени и описания инструмента."""
        assert sql_tool.name == "sql_query"
        assert "SQL" in sql_tool.description
        assert "PostgreSQL" in sql_tool.description
    
    def test_extract_sql(self, sql_tool):
        """Тест 2: Проверка извлечения SQL из ответа LLM."""
        test_cases = [
            ("SELECT * FROM customers", "SELECT * FROM customers"),
            ("```sql\nSELECT * FROM customers\n```", "SELECT * FROM customers"),
            ("```\nSELECT * FROM customers\n```", "SELECT * FROM customers"),
            ("SELECT * FROM customers;", "SELECT * FROM customers"),
            ("Some text SELECT * FROM customers", "SELECT * FROM customers"),
        ]
        
        for input_text, expected in test_cases:
            result = sql_tool._extract_sql(input_text)
            assert result == expected
    
    def test_check_sql_is_safe(self, sql_tool):
        """Тест 3: Проверка безопасности SQL запросов."""
        # Безопасные запросы
        sql_tool._check_sql_is_safe("SELECT * FROM customers")
        sql_tool._check_sql_is_safe("SELECT id, name FROM customers WHERE age > 18")
        sql_tool._check_sql_is_safe("SELECT * FROM customers WHERE city = 'Москва'")
        
        # Опасные запросы
        with pytest.raises(UnsafeSQLError):
            sql_tool._check_sql_is_safe("DROP TABLE customers")
        
        with pytest.raises(UnsafeSQLError):
            sql_tool._check_sql_is_safe("DELETE FROM customers WHERE id = 1")
        
        with pytest.raises(UnsafeSQLError):
            sql_tool._check_sql_is_safe("UPDATE customers SET age = 30")
        
        with pytest.raises(UnsafeSQLError):
            sql_tool._check_sql_is_safe("INSERT INTO customers VALUES (1, 'Test')")
    
    def test_forbidden_keywords(self, sql_tool):
        """Тест 4: Проверка списка запрещенных ключевых слов."""
        forbidden = sql_tool.FORBIDDEN_KEYWORDS
        assert "DROP" in forbidden
        assert "DELETE" in forbidden
        assert "UPDATE" in forbidden
        assert "INSERT" in forbidden
        assert "ALTER" in forbidden
        assert "TRUNCATE" in forbidden
        assert "GRANT" in forbidden
        assert "REVOKE" in forbidden
        assert "CREATE" in forbidden
    
    @patch('llm_agent.tool_sql_query.SQLQueryTool._make_api_request')
    def test_generate_sql_mock(self, mock_request, sql_tool):
        """Тест 5: Генерация SQL с моком."""
        mock_request.return_value = {
            "choices": [{
                "message": {
                    "content": "```sql\nSELECT * FROM customers WHERE city = 'Москва'\n```"
                }
            }]
        }
        
        # Мокаем get_schema_description
        sql_tool.get_schema_description = MagicMock(return_value="Table: customers\n  Columns: id, name, city")
        
        sql = sql_tool._generate_sql("Покажи всех клиентов из Москвы")
        assert "SELECT" in sql.upper()
        assert "customers" in sql.lower()
        assert "москва" in sql.lower()
    
    def test_use_method(self, sql_tool):
        """Тест 6: Проверка метода use."""
        with patch.object(sql_tool, '_generate_sql') as mock_generate:
            with patch.object(sql_tool, '_execute_sql') as mock_execute:
                mock_generate.return_value = "SELECT * FROM customers"
                mock_execute.return_value = [{"id": 1, "name": "Test"}]
                
                result = sql_tool.use("покажи всех")
                assert "SQL запрос" in result
                assert "Test" in result
    
    def test_use_method_error(self, sql_tool):
        """Тест 7: Проверка обработки ошибок в use."""
        with patch.object(sql_tool, '_generate_sql') as mock_generate:
            mock_generate.side_effect = SQLGenerationError("Test error")
            
            result = sql_tool.use("неправильный запрос")
            assert "Ошибка" in result
    
    def test_sql_generation_error(self):
        """Тест 8: Проверка исключения SQLGenerationError."""
        with pytest.raises(SQLGenerationError):
            raise SQLGenerationError("Test error")
    
    def test_unsafe_sql_error(self):
        """Тест 9: Проверка исключения UnsafeSQLError."""
        with pytest.raises(UnsafeSQLError):
            raise UnsafeSQLError("Test error")
    
    def test_ollama_connection(self, sql_tool):
        """Тест 10: Проверка соединения с Ollama."""
        # Мокаем requests.get
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            result = sql_tool.test_ollama_connection()
            assert result is True
        
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Exception("Connection error")
            result = sql_tool.test_ollama_connection()
            assert result is False