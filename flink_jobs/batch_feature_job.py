import argparse
from pathlib import Path

from pyflink.table import EnvironmentSettings, TableEnvironment


def parse_args():
    parser = argparse.ArgumentParser(description="Build one batch of market features with PyFlink.")
    parser.add_argument("--raw-file", required=True, help="Shared path visible to JobManager and TaskManager.")
    parser.add_argument("--output-dir", required=True, help="Shared staging directory visible to TaskManager.")
    return parser.parse_args()


def file_uri(path):
    return Path(path).resolve().as_uri()


def escape_sql(value):
    return value.replace("'", "''")


def main():
    args = parse_args()
    raw_path = Path(args.raw_file)
    output_dir = Path(args.output_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file is not visible to the Flink client: {raw_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o777)

    settings = EnvironmentSettings.new_instance().in_batch_mode().build()
    table_env = TableEnvironment.create(settings)
    configuration = table_env.get_config().get_configuration()
    configuration.set_string("parallelism.default", "1")

    raw_uri = escape_sql(file_uri(raw_path))
    output_uri = escape_sql(file_uri(output_dir))
    table_env.execute_sql(
        f"""
        CREATE TEMPORARY TABLE raw_market_data (
            `timestamp` BIGINT,
            `open` DOUBLE,
            `high` DOUBLE,
            `low` DOUBLE,
            `close` DOUBLE,
            `volume` DOUBLE,
            run_id STRING,
            symbol STRING,
            market STRING,
            timeframe STRING
        ) WITH (
            'connector' = 'filesystem',
            'path' = '{raw_uri}',
            'format' = 'csv'
        )
        """
    )
    table_env.execute_sql(
        f"""
        CREATE TEMPORARY TABLE staged_features (
            `timestamp` BIGINT,
            datetime_utc TIMESTAMP_LTZ(3),
            symbol STRING,
            market STRING,
            timeframe STRING,
            run_id STRING,
            `open` DOUBLE,
            `high` DOUBLE,
            `low` DOUBLE,
            `close` DOUBLE,
            `volume` DOUBLE,
            ma_5 DOUBLE,
            return_1m DOUBLE
        ) WITH (
            'connector' = 'filesystem',
            'path' = '{output_uri}',
            'format' = 'csv'
        )
        """
    )
    result = table_env.execute_sql(
        """
        INSERT INTO staged_features
        SELECT
            `timestamp`,
            TO_TIMESTAMP_LTZ(`timestamp`, 3),
            symbol,
            market,
            timeframe,
            run_id,
            `open`,
            `high`,
            `low`,
            `close`,
            `volume`,
            AVG(`close`) OVER (
                ORDER BY `timestamp`
                ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
            ) AS ma_5,
            COALESCE(
                `close` / LAG(`close`) OVER (ORDER BY `timestamp`) - 1.0,
                0.0
            ) AS return_1m
        FROM raw_market_data
        WHERE `open` > 0 AND `high` > 0 AND `low` > 0 AND `close` > 0 AND `volume` >= 0
        """
    )
    result.wait()
    print(f"Flink feature job completed: {output_dir}")


if __name__ == "__main__":
    main()
