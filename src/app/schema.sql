-- Run once in the Supabase SQL editor to create the complaints table.

create table if not exists public.complaints (
  tracking_id    text primary key,
  submitted_at   timestamptz not null default now(),
  complaint_text text not null,
  team_id        text not null,
  team_name      text not null,
  confidence     real not null,
  status         text not null default 'Received'
);

-- The app connects with the anon (publishable) key, which only ever lives in
-- Streamlit secrets on the server. These policies let it add and look up
-- complaints; it cannot update or delete them.
alter table public.complaints enable row level security;

create policy "app can insert complaints"
  on public.complaints for insert to anon with check (true);

create policy "app can read complaints"
  on public.complaints for select to anon using (true);
