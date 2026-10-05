-- =====================================================================
-- Migracion: Aislamiento vectorial multi-tenant (Requisito 1)
-- Agrega la dimension logica `coleccion_id` a normativa_bancaria y crea
-- la funcion RPC `match_normativa_coleccion` que restringe la busqueda
-- vectorial a una unica coleccion tematica.
--
-- Es idempotente: puede ejecutarse varias veces sin efectos secundarios.
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Columna de tenant
-- ---------------------------------------------------------------------
-- DEFAULT 'asfi_rnsf' preserva las filas ya ingeridas: pasan a quedar
-- etiquetadas en la coleccion normativa bancaria sin perder contenido.
alter table normativa_bancaria
  add column if not exists coleccion_id varchar(50);

update normativa_bancaria
   set coleccion_id = 'asfi_rnsf'
 where coleccion_id is null;

alter table normativa_bancaria
  alter column coleccion_id set not null;

-- El DEFAULT deja de ser necesario: toda inserta futura debe declarar
-- su coleccion de forma explicita (fail-fast en lugar de colarse
-- silenciosamente en la coleccion por defecto).
alter table normativa_bancaria
  alter column coleccion_id drop default;

-- ---------------------------------------------------------------------
-- 2. Indice B-Tree por coleccion (filtrado determinista del tenant)
-- ---------------------------------------------------------------------
-- El B-Tree resuelve `coleccion_id = p_coleccion_id` en indice, de modo
-- que el planner descarta filas de otros tenants antes de calcular
-- distancias coseno.
create index if not exists normativa_bancaria_coleccion_idx
  on normativa_bancaria using btree (coleccion_id);

-- Indice compuesto para el patron real de uso: filtrar por coleccion y
-- ordenar por similitud. Reduce el numero de tuplas candidatas que llegan
-- a la comparacion vectorial.
create index if not exists normativa_bancaria_coleccion_embedding_idx
  on normativa_bancaria using btree (coleccion_id, documento_origen);

-- ---------------------------------------------------------------------
-- 3. RPC de busqueda aislada por coleccion
-- ---------------------------------------------------------------------
-- Los parametros se prefijan con `p_` a proposito: en PL/pgSQL un
-- parametro sin prefijar que coincida con el nombre de una columna genera
-- ambiguedad de ambito y puede romper silenciosamente el filtro.
create or replace function match_normativa_coleccion(
  query_embedding  vector(768),
  match_threshold  float,
  match_count      int,
  p_coleccion_id   varchar(50)
)
returns table (
  id               bigint,
  documento_origen text,
  organismo        text,
  tipo_norma       text,
  jerarquia        text,
  articulo_ref     text,
  contenido        text,
  coleccion_id     varchar(50),
  similarity       float
)
language plpgsql
stable
set search_path = public
as $$
begin
  return query
    select
      nb.id,
      nb.documento_origen,
      nb.organismo,
      nb.tipo_norma,
      nb.jerarquia,
      nb.articulo_ref,
      nb.contenido,
      nb.coleccion_id,
      1 - (nb.embedding <=> query_embedding) as similarity
    from normativa_bancaria nb
    where nb.coleccion_id = p_coleccion_id
      and 1 - (nb.embedding <=> query_embedding) > match_threshold
    order by nb.embedding <=> query_embedding
    limit match_count;
end;
$$;

comment on function match_normativa_coleccion(vector, float, int, varchar) is
  'Busqueda vectorial aislada por coleccion. El filtro coleccion_id se '
  'evalua sobre el indice B-Tree, garantizando cero contaminacion cruzada.';

-- ---------------------------------------------------------------------
-- 4. RPC auxiliar: inventario de colecciones
-- ---------------------------------------------------------------------
create or replace function listar_colecciones()
returns table (
  coleccion_id      varchar(50),
  total_fragmentos  bigint
)
language plpgsql
stable
set search_path = public
as $$
begin
  return query
    select nb.coleccion_id, count(*)::bigint
      from normativa_bancaria nb
     group by nb.coleccion_id
     order by nb.coleccion_id;
end;
$$;