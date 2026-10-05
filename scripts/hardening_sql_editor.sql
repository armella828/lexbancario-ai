-- =====================================================================
-- Endurecimiento de `ejecutar_sql`
--
-- La funcion es SECURITY DEFINER: ejecuta SQL con privilegios del
-- propietario, no de quien la llama. Por defecto PostgreSQL concede
-- EXECUTE a PUBLIC, lo que significa que cualquier rol con una API key
-- de Supabase (anon, authenticated) podria ejecutar DDL arbitrario en la
-- base: crear usuarios, leer otras tablas, borrar el vector store.
--
-- Aqui se restringe EXECUTE al unico rol que el proyecto usa para
-- migrar: service_role (la service role key).
-- =====================================================================

do $$
begin
  -- Se revoca al grupo PUBLIC y a los roles expuestos por PostgREST.
  revoke execute on function public.ejecutar_sql(text)
    from public, anon, authenticated;

  -- Se concede explicitamente solo a service_role.
  grant execute on function public.ejecutar_sql(text)
    to service_role;
end;
$$;