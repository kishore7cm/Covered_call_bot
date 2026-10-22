# Integration tests (real Postgres required)

`test_logger_postgres.py` exercises logger.py's DB-backed functions
(schema migration, `lots` CRUD, the CTE-based roll-count query, etc.)
against a REAL Postgres — everywhere else in this repo, these functions
are monkeypatched, so SQL/schema bugs are otherwise untested.

These tests are **skipped automatically** unless `TEST_DATABASE_URL` is
set, and are not part of the default `pytest tests/` run for that reason
(no Postgres is required to run the rest of the suite).

**Never point `TEST_DATABASE_URL` at a real/production database** —
`init_schema()` creates tables and these tests `TRUNCATE` data in
whatever database the DSN names.

## Running against a throwaway local Postgres (no Docker required)

```bash
brew install postgresql@16   # one-time

PGDATA=/tmp/ccbot_pg_throwaway_data
SOCKDIR=/tmp/ccbot_pg_throwaway_sock
rm -rf "$PGDATA" "$SOCKDIR" && mkdir -p "$SOCKDIR"
/opt/homebrew/opt/postgresql@16/bin/initdb -D "$PGDATA" -U testuser
/opt/homebrew/opt/postgresql@16/bin/pg_ctl -D "$PGDATA" -l "$PGDATA/logfile" \
  -o "-p 5544 -k $SOCKDIR -h localhost" start
/opt/homebrew/opt/postgresql@16/bin/createdb -h localhost -p 5544 -U testuser ccbot_test

export TEST_DATABASE_URL="postgresql://testuser@localhost:5544/ccbot_test"
python3 -m pytest tests/integration/ -q

# Tear down when done -- fully disposable, nothing persists.
/opt/homebrew/opt/postgresql@16/bin/pg_ctl -D "$PGDATA" stop
rm -rf "$PGDATA" "$SOCKDIR"
```

## Running against Docker instead

```bash
docker run --rm -d --name ccbot-pg-throwaway \
  -e POSTGRES_USER=testuser -e POSTGRES_DB=ccbot_test \
  -e POSTGRES_HOST_AUTH_METHOD=trust \
  -p 5544:5432 postgres:16

export TEST_DATABASE_URL="postgresql://testuser@localhost:5544/ccbot_test"
python3 -m pytest tests/integration/ -q

docker stop ccbot-pg-throwaway
```
