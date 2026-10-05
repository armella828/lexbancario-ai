-- =====================================================================
-- Funciones de introspeccion del esquema (verificacion y reporte)
-- A diferencia de `ejecutar_sql`, estas SI devuelven filas consultables
-- a traves de la API REST, porque declaran su propio tipo de retorno.
-- =====================================================================

create or replace function verificar_indices(p_tabla text default 'normativa_bancaria')
returns table (
  nombre_indice   text,
  definicion      text
)
language sql
stable
set search_path = public
as $$
  select i.indexname, i.indexdef
    from pg_indexes i
   where i.tablename = p_tabla
   order by i.indexname;
$$;

create or replace function verificar_columnas(p_tabla text default 'normativa_bancaria')
returns table (
  columna        text,
  tipo           text,
  es_nulo        text,
  valor_default  text
)
language sql
stable
set search_path = public
as $$
  select
    c.column_name,
    c.data_type,
    case when c.is_nullable = 'NO' then 'NO' else 'SI' end,
    coalesce(c.column_default, '(ninguno)')
  from information_schema.columns c
  where c.table_name = p_tabla
  order by c.ordinal_position;
$$;