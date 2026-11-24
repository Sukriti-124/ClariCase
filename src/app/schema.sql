-- Run once in the Supabase SQL editor to create the complaints table.

create table if not exists public.complaints (
  tracking_id    text primary key,
  submitted_at   timestamptz not null default now(),
  complaint_text text not null,
  team_id        text not null,
  team_name      text not null,
  confidence     real not null,
  status         text not null default 'Received',  -- Routed / Under review / Assigned
  issue_id       text,
  issue_label    text,
  route          text                               -- auto / review
);


alter table public.complaints add column if not exists issue_id    text;
alter table public.complaints add column if not exists issue_label text;
alter table public.complaints add column if not exists route       text;

alter table public.complaints add column if not exists decision_record   jsonb;
alter table public.complaints add column if not exists explanation       jsonb;
alter table public.complaints add column if not exists explanation_audit jsonb;


-- Admin review: set when a "needs review" complaint is assigned to a team.
alter table public.complaints add column if not exists model_team_id   text;
alter table public.complaints add column if not exists model_team_name text;
alter table public.complaints add column if not exists reviewed_at     timestamptz;
alter table public.complaints add column if not exists review_note     text;


alter table public.complaints enable row level security;


