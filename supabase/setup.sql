-- VisionFlux shared review schema
-- Run once in Supabase Dashboard > SQL Editor.

create extension if not exists pgcrypto;

create table if not exists public.visionflux_projects (
  id uuid primary key default gen_random_uuid(),
  slug text not null unique,
  name text not null,
  created_at timestamptz not null default now()
);

create table if not exists public.visionflux_images (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references public.visionflux_projects(id) on delete cascade,
  image_name text not null,
  image_hash text not null,
  storage_path text not null,
  uploaded_by text,
  status text not null default 'pending' check (status in ('pending','in_progress','done','error')),
  locked_by text,
  locked_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  unique(project_id, image_hash)
);

create table if not exists public.visionflux_reviews (
  id uuid primary key default gen_random_uuid(),
  image_id uuid not null unique references public.visionflux_images(id) on delete cascade,
  snapshot jsonb not null default '{}'::jsonb,
  revision integer not null default 0,
  updated_by text,
  status text not null default 'in_progress' check (status in ('in_progress','done')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.visionflux_artifacts (
  id uuid primary key default gen_random_uuid(),
  image_id uuid not null references public.visionflux_images(id) on delete cascade,
  kind text not null,
  storage_path text not null,
  created_by text,
  created_at timestamptz not null default now()
);

create index if not exists visionflux_images_project_idx on public.visionflux_images(project_id);
create index if not exists visionflux_images_status_idx on public.visionflux_images(status);
create index if not exists visionflux_reviews_image_idx on public.visionflux_reviews(image_id);
create index if not exists visionflux_artifacts_image_idx on public.visionflux_artifacts(image_id);

alter table public.visionflux_projects enable row level security;
alter table public.visionflux_images enable row level security;
alter table public.visionflux_reviews enable row level security;
alter table public.visionflux_artifacts enable row level security;

-- No anon/authenticated policies are intentionally created. The Streamlit backend
-- uses a service-role key stored in Streamlit Secrets. Never put that key in GitHub
-- or browser JavaScript.

insert into storage.buckets (id, name, public)
values ('visionflux-images', 'visionflux-images', false)
on conflict (id) do update set public = excluded.public;

insert into storage.buckets (id, name, public)
values ('visionflux-results', 'visionflux-results', false)
on conflict (id) do update set public = excluded.public;

create or replace function public.visionflux_acquire_lock(
  p_image_id uuid,
  p_worker text,
  p_timeout_minutes integer default 30
)
returns table(acquired boolean, locked_by text, locked_at timestamptz, status text)
language plpgsql
security definer
set search_path = public
as $$
declare
  current_row public.visionflux_images%rowtype;
begin
  if coalesce(trim(p_worker), '') = '' then
    raise exception 'worker name is required';
  end if;

  select * into current_row
  from public.visionflux_images
  where id = p_image_id
  for update;

  if not found then
    return query select false, null::text, null::timestamptz, 'missing'::text;
    return;
  end if;

  if current_row.locked_by is null
     or current_row.locked_by = p_worker
     or current_row.locked_at < now() - make_interval(mins => greatest(p_timeout_minutes, 5)) then
    update public.visionflux_images
       set locked_by = p_worker,
           locked_at = now(),
           status = case when status = 'done' then status else 'in_progress' end,
           updated_at = now()
     where id = p_image_id
     returning visionflux_images.locked_by, visionflux_images.locked_at, visionflux_images.status
      into current_row.locked_by, current_row.locked_at, current_row.status;
    return query select true, current_row.locked_by, current_row.locked_at, current_row.status;
  else
    return query select false, current_row.locked_by, current_row.locked_at, current_row.status;
  end if;
end;
$$;

create or replace function public.visionflux_release_lock(
  p_image_id uuid,
  p_worker text,
  p_completed boolean default false
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
  update public.visionflux_images
     set locked_by = null,
         locked_at = null,
         status = case when p_completed then 'done' else status end,
         updated_at = now()
   where id = p_image_id
     and locked_by = p_worker;
end;
$$;

revoke all on function public.visionflux_acquire_lock(uuid,text,integer) from public, anon, authenticated;
revoke all on function public.visionflux_release_lock(uuid,text,boolean) from public, anon, authenticated;
grant execute on function public.visionflux_acquire_lock(uuid,text,integer) to service_role;
grant execute on function public.visionflux_release_lock(uuid,text,boolean) to service_role;
