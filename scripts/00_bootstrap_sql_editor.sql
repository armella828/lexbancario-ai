-- =====================================================================
-- BOOTSTRAP UNICO - pegar en el SQL Editor de Supabase (una sola vez)
-- Dashboard > Projecto > SQL Editor > New query > Run
--
-- Crea la funcion `ejecutar_sql`, que permite aplicar migraciones DDL
-- desde Python. Supabase no expone DDL por la API REST, asi que esta
-- funcion es el puente necesario para que `scripts/aplicar_sql.py`
-- funcione y las migraciones sean reproducibles.
-- =====================================================================

create or replace function public.ejecutar_sql(sql_texto text)
returns text
language plpgsql
security definer
set search_path = public
as $$
begin
  execute sql_texto;
  return 'ok';
end;
$$;

-- Verificacion: debe responder 'ok'
select public.ejecutar_sql('select 1') as bootstrap_ok;