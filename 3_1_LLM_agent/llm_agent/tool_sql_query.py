# llm_agent/tool_sql_query.py

import json
import re
from typing import Any, Dict, List, Optional

import psycopg2
import psycopg2.extras
import requests
from decouple import config

class SQLGenerationError(Exception):
    """Ошибка при генерации SQL-запроса с помощью LLM."""
    pass


class UnsafeSQLError(Exception):
    """Сгенерированный SQL-запрос содержит потенциально опасную операцию."""
    pass

class SQLQueryTool:
    """Инструмент для выполнения запросов SQL."""
    
    name = "sql_query"
    description = "Выполнение SQL-запросов к локальной базе SQL (PostgreSQL) на основе текстового описания"
    
    FORBIDDEN_KEYWORDS = (
        "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
        "TRUNCATE", "GRANT", "REVOKE", "CREATE", "DROP",
    )
    
    def __init__(
            self,
            model: str = "tngtech/deepseek-r1t2-chimera",
            local: bool = False,
            ollama_base_url: str = config("OLLAMA_BASE_URL", default="http://localhost:11434"),
            ollama_model: str = config("OLLAMA_MODEL", default="qwen3.5:0.8b"),
            pg_host: str = None,
            pg_port: str = None,
            pg_db: str = None,
            pg_user: str = None,
            pg_password: str = None,
            allow_write: bool = False,
            timeout: int = 60,
            auto_create_schema: bool = False,
        ) -> None:
            """ Инициализирует SQL-инструмент.
            
            Args:
                model (str): Название модели для OpenRouter.
                local (bool): Если True, используется локальный Ollama.
                ollama_base_url (str): Базовый URL Ollama API.
                ollama_model (str): Название локальной модели Ollama.
                pg_host (str): Хост PostgreSQL.
                pg_port (str): Порт PostgreSQL.
                pg_db (str): Название базы данных.
                pg_user (str): Пользователь PostgreSQL.
                pg_password (str): Пароль PostgreSQL.
                allow_write (bool): Разрешить изменяющие SQL-запросы.
                timeout (int): Таймаут HTTP-запроса. """
            
            # Настройка LLM
            
            self.local = local
            self.ollama_base_url = ollama_base_url
            self.ollama_model = ollama_model
            self.timeout = timeout
            
            if not self.local:
                self.api_key = config('OPENROUTER_API_KEY')
                self.url = "https://openrouter.ai/api/v1/chat/completions"
                self.model = model
            else:
                self.api_key = None
                self.url = f"{self.ollama_base_url}/v1/chat/completions"
                self.model = ollama_model
            
            # Значения по умолчанию читаются из .env
            self.pg_host = pg_host or config("PG_HOST", default="localhost")
            self.pg_port = pg_port or config("PG_PORT", default="5432")
            self.pg_db = pg_db or config("PG_DATABASE", default="postgres")
            self.pg_user = pg_user or config("PG_USER", default="postgres")
            self.pg_password = pg_password or config("PG_PASSWORD", default="postgres")
    
            self.allow_write = allow_write
            self.connection = None
            try:
                self._ensure_connection()
                if auto_create_schema:
                    self.create_sample_schema()
            except ConnectionError as e:
                print(f"[SQLQueryTool] PostgreSQL недоступен, работаю без БД: {e}")
    
            # self.connection = psycopg2.connect(
            #     host=self.pg_host,
            #     port=self.pg_port,
            #     dbname=self.pg_db,
            #     user=self.pg_user,
            #     password=self.pg_password,
            # )
            
    def _connect(self):
        """Устанавливает соединение с PostgreSQL."""
        try:
            self.connection = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                dbname=self.pg_db,
                user=self.pg_user,
                password=self.pg_password,
                client_encoding='UTF8',
            )
            print(f"Подключено к PostgreSQL: {self.pg_host}:{self.pg_port}/{self.pg_db}")
        except Exception as e:
            error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            raise ConnectionError(f"Не удалось подключиться к PostgreSQL: {error_msg}")
        
    def _ensure_connection(self):
        """Подключается к PostgreSQL при первом обращении, если соединения ещё нет."""
        if self.connection is None:
            self._connect()

    def create_sample_schema(self) -> None:
        """Создаёт демонстрационную таблицу customers с тестовыми данными."""
        self._ensure_connection()
        with self.connection.cursor() as cur:
            cur.execute(
                """
                DROP TABLE IF EXISTS customers CASCADE;
                CREATE TABLE customers (
                    id SERIAL PRIMARY KEY,
                    first_name TEXT NOT NULL,
                    second_name TEXT NOT NULL,
                    city TEXT NOT NULL,
                    age INTEGER,
                    email TEXT,
                    phone TEXT
                );
                """
            )
            cur.executemany(
                """INSERT INTO customers (first_name, second_name, city, age, email, phone)
                    VALUES (%s, %s, %s, %s, %s, %s)""",
                [
                    ("Иван", "Петров", "Москва", 34, "ivan@mail.ru", "+7-999-123-45-67"),
                    ("Анна", "Смирнова", "Санкт-Петербург", 28, "anna@mail.ru", "+7-999-234-56-78"),
                    ("Олег", "Кузнецов", "Москва", 41, "oleg@mail.ru", "+7-999-345-67-89"),
                    ("Мария", "Иванова", "Казань", 25, "maria@mail.ru", "+7-999-456-78-90"),
                    ("Дмитрий", "Соколов", "Москва", 35, "dmitry@mail.ru", "+7-999-567-89-01"),
                ],
            )
        self.connection.commit()
        print("Создана демонстрационная таблица customers с 5 записями")
        
    def get_schema_description(self) -> str:
        """Возвращает текстовое описание схемы БД."""
        self._ensure_connection()
        with self.connection.cursor() as cur:
            cur.execute(
                """
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position;
                """
            )
            rows = cur.fetchall()

        schema: Dict[str, List[str]] = {}
        for table_name, column_name, data_type in rows:
            schema.setdefault(table_name, []).append(f"{column_name} ({data_type})")

        if not schema:
            return "В базе данных нет таблиц."

        return "\n".join(
            f"Таблица {table}: " + ", ".join(columns)
            for table, columns in schema.items()
        )
    
    def _generate_sql(self, question: str) -> str:
        """Генерирует SQL-запрос на основе вопроса пользователя."""
        schema = self.get_schema_description()
        system_prompt = (
            "You are an assistant that translates natural language questions "
            "into SQL queries for a PostgreSQL database.\n\n"
            f"Database schema:\n{schema}\n\n"
            "Strict rules:\n"
            "1. Return ONLY the SQL query itself, as a single statement, with no explanation, "
            "no markdown formatting, and no trailing semicolon.\n"
            "2. Use ONLY standard PostgreSQL syntax.\n"
            "3. Use only the tables and columns listed in the schema above.\n"
            "4. Always use proper SQL syntax with correct quotes and escaping.\n"
            "5. For Russian text, use single quotes: 'Москва'"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        }
        
        if self.local:
            payload["stream"] = False
            payload["think"] = False
        
        try:
            response_data = self._make_api_request(payload)
            llm_text = response_data["choices"][0]["message"]["content"]
            
            if not llm_text:
                raise SQLGenerationError("LLM вернула пустой ответ.")
            
            sql = self._extract_sql(llm_text)
            return sql
        
        except (KeyError, json.JSONDecodeError, SQLGenerationError) as e:
            raise SQLGenerationError(f"Не удалось получить SQL от LLM: {e}") from e
        
    @staticmethod
    def _extract_sql(text: str) -> str:
        """Извлекает чистый SQL из ответа модели."""
        # Убираем маркеры markdown
        match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        sql = match.group(1) if match else text
        
        sql = sql.strip().rstrip(";").strip()
        
        # Проверяем, что запрос начинается с SELECT
        if not sql.upper().startswith("SELECT"):
            # Пробуем найти SELECT в тексте
            select_match = re.search(r"(SELECT.*?)(?:;|$)", sql, re.DOTALL | re.IGNORECASE)
            if select_match:
                sql = select_match.group(1).strip()
        
        return sql
    
    def _check_sql_is_safe(self, sql: str) -> None:
        """Проверяет SQL на безопасность."""
        if self.allow_write:
            return
        
        upper_sql = sql.upper()
        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(rf"\b{keyword}\b", upper_sql):
                raise UnsafeSQLError(
                    f"Запрос содержит запрещённую операцию '{keyword}'. "
                    "По умолчанию разрешены только SELECT-запросы."
                )
                
    def _execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        """Проверяет и выполняет SQL-запрос."""
        self._ensure_connection()
        sql = self._extract_sql(sql)
        self._check_sql_is_safe(sql)
        
        with self.connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            if sql.strip().upper().startswith("SELECT"):
                rows = [dict(row) for row in cur.fetchall()]
            else:
                rows = []
        self.connection.commit()
        return rows
    
    def _make_api_request(self, payload: Dict, headers: Optional[Dict] = None) -> Dict:
        """Универсальный метод для отправки запросов к API."""
        if headers is None:
            headers = {}
        
        if not self.local:
            headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        else:
            headers["Content-Type"] = "application/json"
        
        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")
    
    def use(self, query: str) -> str:
        """
        Принимает строку с запросов на человеческом языке и возвращает результат
        
        Args:
            expression (str): Выражение для запроса, например, "Сколько клиентов живет в Москве".
            
        Returns:
            str: Строка с результатом или сообщение об ошибке
        """
        try:
            sql = self._generate_sql(query)
            rows = self._execute_sql(sql)
            result_str = f"SQL запрос: {sql}\n\n"
            if rows:
                result_str += f"Найдено записей: {len(rows)}\n"
                result_str += json.dumps(rows, ensure_ascii=False, indent=2)
            else:
                result_str += "Записей не найдено."
                
            return result_str
        except SQLGenerationError as e:
            return f"Ошибка генерации SQL: {e}"
        except UnsafeSQLError as e:
            return f"Ошибка безопасности: {e}"
        except Exception as e:
            return f"Ошибка выполнения SQL запроса: {e}"
        
    def run(self, question: str) -> Dict[str, Any]:
        """Полный цикл: вопрос на естественном языке -> SQL -> результат."""
        sql = self._generate_sql(question)
        rows = self._execute_sql(sql)
        return {"question": question, "sql": sql, "rows": rows}
    
    def close(self) -> None:
        """Закрывает соединение с БД."""
        if self.connection:
            self.connection.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def test_ollama_connection(self) -> bool:
        """Тестирует соединение с локальным Ollama сервером."""
        if not self.local:
            return False
        
        try:
            test_url = f"{self.ollama_base_url}/v1/models"
            response = requests.get(test_url, timeout=5)
            return response.status_code == 200
        except:
            return False
