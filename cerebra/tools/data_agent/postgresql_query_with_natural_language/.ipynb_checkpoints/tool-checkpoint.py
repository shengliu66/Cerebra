# cerebra/tools/data_tools/load_pickle_data/tool.py

from typing import Dict, Tuple, Any
import sqlalchemy
from sqlalchemy import create_engine, text, inspect
import pandas as pd

from cerebra.tools.base import BaseTool
from cerebra.engine.factory import create_llm_engine
from cerebra.agents.modules.formatters import SqlQuery, TableAndColumnDescription
from cerebra.utils.dataset import Dataset
from cerebra.agents.modules.utils import safe_parse_dict


class PostgreSqlQueryWithNaturalLanguageTool(BaseTool):
    require_llm_engine = True

    def __init__(
        self,
        model_string="gpt-4o-mini",
        ):
        super().__init__(
            tool_name="PostgreSqlQueryWithNaturalLanguageTool",
            tool_description="Given a PostgreSQL database, generate the SQL query schema for a natural language query request",
            tool_version="1.0.0",
            input_types={
                "pg_database_url": "str - URL to access the PostgreSQL database",
                "natural_language_query": "str - The natural language query to be answered from the database",
            },
            output_type="dict - A dictionary containing queried data, and metadata",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(pg_database_url="postgresql+psycopg2://example_user@server_location:server_port/postgres"), natural_language_query="Give me all patients diagnosed with dementia in the past 6 months.")',
                    "description": "Query the PostgreSQL database to find all patients diagnosed with dementia in the past 6 months.",
                },
                {
                    "command": 'execution = tool.execute(pg_database_url="postgresql+psycopg2://alice@localhost:120.0.0.1:7821/pg_database"), natural_language_query="Find all MPR MRI scans taken place in the past 200 days.")',
                    "description": "Query the PostgreSQL database to find MPR MRI scans in the past 200 days.",
                },
                {
                    "command": 'execution = tool.execute(pg_database_url="postgresql+psycopg2://bobby@cn-1020:6986/pg_sql_server"), natural_language_query="Give me all FLAIR MRI scans for the patients who have ADHD in the past 5 years.")',
                    "description": "Query the PostgreSQL database to find all FLAIR MRI scans for the patients who have ADHD in the past 5 years.",
                },
            ],
            user_metadata={
                "limitations": [],
                "best_practices": [
                    """Uppercase columns of a table should use double quote (e.g. table1."COLUMN1")"""
                ],
            }
        )

        print(f"Initializing Python_Code_Generator_Tool with model_string: {model_string}")
        self.llm_engine = create_llm_engine(
            model_string=model_string,
            is_multimodal=False
            ) if model_string else None

    @staticmethod
    def _execute_postgresql_query(
        sql_query: str,
        postgresql_engine: sqlalchemy.engine.base.Engine,
        ) -> pd.DataFrame:
        """
        Execute a raw SQL query using a SQLAlchemy PostgreSQL engine and return the result as a DataFrame.

        Args:
            sql_query (str): The SQL query string to be executed.
            postgresql_engine (sqlalchemy.engine.base.Engine): A SQLAlchemy engine connected to the PostgreSQL database.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the result of the SQL query execution.
        """
        with postgresql_engine.connect() as connection:
            result = connection.execute(text(sql_query))
            rows = result.fetchall()
            columns = result.keys()
            df = pd.DataFrame(rows, columns=columns)

        return df

    def generate_table_and_column_description(
        self,
        dataframe: pd.DataFrame,
        sql_query: str,
        natural_language_query: str,
        postgresql_engine: sqlalchemy.engine.base.Engine,
        ) -> Tuple[str, Dict[str, str]]:
        """
        Generate description for result table and columns
        Args:
            dataframe (pd.DataFrame): The DataFrame resulting from the executed SQL query.
            sql_query (str): The SQL query used to generate the DataFrame.
            natural_language_query (str): The user's original natural language query.
            postgresql_engine (sqlalchemy.engine.base.Engine): SQLAlchemy engine connected to the PostgreSQL database.
        Returns:
            (str) Table description
            (Dict[str, str]) Descriptions for columns
        """

        schema_text = self._generate_pg_schema_for_llm(postgresql_engine=postgresql_engine)
        dataframe_columns_str = f"""[{', '.join(dataframe.columns)}]"""
        prompt = f"""
Given the following PostgreSQL schema:

{schema_text}

Find the most appropriate table and column descriptions for the dataframe, where the columns are:

{dataframe_columns_str}

Which is generated for this natural language question:

"{natural_language_query}"

And by this SQL query:

{sql_query}
"""
        llm_output = self.llm_engine(
            prompt=prompt,
            response_format=TableAndColumnDescription,
            )
        table_description = llm_output.table_description
        column_description_dict = safe_parse_dict(llm_output.column_description_dict)

        return table_description, column_description_dict

    def generate_query_from_natural_language(
        self,
        natural_language_query: str,
        postgresql_engine,
        ) -> Dict[str, str]:
        """
        Args:
            natural_language_query: Natural language query for database
            postgresql_engine: Engine for postgresql database
        Returns: 
            Dictionary with SQL query and explanation of the query
        """

        schema_text = self._generate_pg_schema_for_llm(postgresql_engine=postgresql_engine)
        prompt = f"""
Given the following PostgreSQL schema:

{schema_text}

Translate the following natural language question into a valid SQL query:

"{natural_language_query}"

Notes:
1. Whenever possible, always include pat_mrn_id and accession_num
2. Uppercase columns of a table should use double quote (e.g. table1."COLUMN1")
"""
        llm_output = self.llm_engine(
            prompt=prompt,
            response_format=SqlQuery,
            )

        return {
            "sql_query": llm_output.sql_query,
            "explanation": llm_output.explanation,
            }

    @staticmethod
    def _generate_pg_schema_for_llm(
        postgresql_engine,
        include_types=True,
        ) -> str:
        """
        Extracts the PostgreSQL schema from a SQLAlchemy engine and formats it
        for LLM prompt input.

        Args:
            postgresql_engine: SQLAlchemy engine connected to PostgreSQL
            include_types: If True, includes column data types

        Returns:
            A formatted string representing the schema
        """
        inspector = inspect(postgresql_engine)
        schema_str = ""

        for table_name in inspector.get_table_names():
            schema_str += f"Table: {table_name}\n"
            schema_str += "Columns:\n"

            for column in inspector.get_columns(table_name):
                col_line = f"  - {column['name']}"
                if include_types:
                    col_line += f" ({column['type']})"
                schema_str += col_line + "\n"

            schema_str += "\n"

        return schema_str.strip()


    def execute(
        self,
        postgresql_database_url: str,
        natural_language_query: str,
        ) -> Dict[str, Any]:
        """
        Args:
            postgresql_database_url: URL to access the PostgreSQL database
            natural_language_query: The natural language query to be answered from the database
        
        Returns:
            Dictionary containing queried data, and metadata
        """

        #TODO: Make sure the engine is runnable
        postgresql_engine = create_engine(postgresql_database_url)

        # Generate SQL query
        sql_query_output = self.generate_query_from_natural_language(
            natural_language_query=natural_language_query,
            postgresql_engine=postgresql_engine,
            )

        # Execute SQL query
        postgresql_query_output_df = self._execute_postgresql_query(
            sql_query=sql_query_output["sql_query"],
            postgresql_engine=postgresql_engine,
            )

        # Generate table and column descriptions for the result table
        table_description, column_description_dict = self.generate_table_and_column_description(
            dataframe=postgresql_query_output_df,
            sql_query=sql_query_output["sql_query"],
            natural_language_query=natural_language_query,
            postgresql_engine=postgresql_engine,
            )

        # Generate Dataset instance
        tool_output_dataset = Dataset(
            dataset=postgresql_query_output_df.to_dict(),
            dataset_type='raw_dataset',
            dataset_description=table_description,
            feature_description=column_description_dict,
            cache_directory="cerebra_cache/",
        )

        return {
            "status": "success",
            "length_records": len(tool_output_dataset),
            "data": tool_output_dataset,
            "postgresql_database_url": postgresql_database_url,
            "natural_language_query": natural_language_query,
            "sql_query": sql_query_output["sql_query"],
            "sql_query_explanation": sql_query_output["explanation"],
        }

    def get_metadata(self):
        """
        Returns the metadata for the PostgreSqlQueryWithNaturalLanguageTool.

        Returns:
            dict: A dictionary containing the tool's metadata.
        """
        metadata = super().get_metadata()
        metadata["require_llm_engine"] = self.require_llm_engine
        return metadata


if __name__ == "__main__":
    # Test command:
    """
        Run the following commands in the terminal to test the script:
        
        cd cerebra/tools/data_agent/postgresql_query_with_natural_language
        python tool.py
    """

    # Example usage of the postgresql_query_with_natural_language
    tool = PostgreSqlQueryWithNaturalLanguageTool()

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(metadata)

    try:
        execution = tool.execute(
            postgresql_database_url="postgresql+psycopg2://example_user@server_location:server_port/postgres",
            natural_language_query="Give me all patients diagnosed with dementia in the past 6 years, with one T1 MRI scan for each patient in the session that the patients are diagnosed.",
        )
        print("\n###Execution Result:", execution)
        print("*" * 50)
        print(execution['data'].get_dataset_info())
    except Exception as e:
        print(f"Unexpected error: {e}")

    print("Done!")
