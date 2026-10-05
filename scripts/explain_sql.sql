-- ============================================================
-- EXPLAIN de consultas, devuelto como texto.
--
-- `EXPLAIN` es una sentencia que devuelve filas, no un valor, asi que la
-- funcion `ejecutar_sql` (que hace `execute`) no sirve para capturarla:
-- hace falta `EXPLAIN (FORMAT TEXT) ...` dentro de plpgsql, acumulando las
-- lineas del plan en un texto.
--
-- Es de solo lectura. Existe para poder publicar evidencia de que los
-- indices se usan de verdad, en vez de suponerlo.
-- ============================================================
create or replace function explicar_consulta(sql_texto text)
returns text
language plpgsql
security definer
set search_path = public
as $$
declare
  fila    record;
  lineas  text := '';
  conteo  int := 0;
begin
  for fila in execute 'explain (analyze, buffers, format text) ' || sql_texto
  loop
    lineas := lineas || fila."QUERY PLAN" || E'\n';
    conteo := conteo + 1;
  end loop;

  if conteo = 0 then
    return '(el plan no devolvio lineas)';
  end if;
  return lineas;
end;
$$;

comment on function explicar_consulta(text) is
  'Ejecuta EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) y devuelve el plan como '
  'texto. Solo lectura: sirve para documentar evidencia de uso de indices.';