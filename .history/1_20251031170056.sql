-- ============================================================
-- 🧾 QUARTERLY SCHEME-WISE AUM, FOLIO, INFLOW, OUTFLOW REPORT
-- Financial Year: April → March
-- ============================================================
-- 1️⃣ Parameters for Quarter Selection
with params as (
    select -- ✅ Update these 2 dates for the current reporting quarter
        date('2025-06-30') as curr_qtr_end,
        -- Current Quarter End (e.g., Q1 FY25–26)
        date('2025-03-31') as prev_qtr_end,
        -- Previous Quarter End (e.g., Q4 FY24–25)
        '125' as fund_code -- Fund code parameter
),
-- ============================================================
-- 2️⃣  Previous Quarter AUM (excluding intra)
-- ============================================================
prev_qtr_aum as (
    select a.scheme,
        a.plan,
        a.fund,
        sum(cast(a.aum as decimal(38, 3))) as openaum
    from standardmodel.aum_meview a
        join params p on a.fund = p.fund_code
    where date(
            from_iso8601_date(
                concat(a.creation_year, '-', a.creation_month, '-01')
            )
        ) <= p.prev_qtr_end
        and a.creation_month in ('03', '06', '09', '12')
        and a.scheme not in (
            select distinct fm_scheme
            from bronze.canara_dbo_mcr_schexclude_ro
        )
        and concat(a.scheme, a.plan) not in (
            select concat(scheme, plan)
            from bronze.global_plan_exclude
            where fund = p.fund_code
        )
    group by a.scheme,
        a.plan,
        a.fund
),
-- ============================================================
-- 3️⃣  Previous Quarter Intra AUM (internal adjustments)
-- ============================================================
prev_qtr_intra as (
    select d.scheme,
        d.plan,
        d.fund,
        sum(cast(d.aum as decimal(38, 3))) as openintraaum
    from standardmodel.aum_dview d
        join params p on d.fund = p.fund_code
    where date(d.batchclosedt) = p.prev_qtr_end
        and d.scheme not in (
            select distinct fm_scheme
            from bronze.canara_dbo_mcr_schexclude_ro
        )
        and concat(d.scheme, d.plan) not in (
            select concat(scheme, plan)
            from bronze.global_plan_exclude
            where fund = p.fund_code
        )
    group by d.scheme,
        d.plan,
        d.fund
),
-- ============================================================
-- 4️⃣  Current Quarter AUM (excluding intra)
-- ============================================================
curr_qtr_aum as (
    select a.scheme,
        a.plan,
        a.fund,
        sum(cast(a.aum as decimal(38, 3))) as closeaum
    from standardmodel.aum_meview a
        join params p on a.fund = p.fund_code
    where date(
            from_iso8601_date(
                concat(a.creation_year, '-', a.creation_month, '-01')
            )
        ) <= p.curr_qtr_end
        and a.creation_month in ('03', '06', '09', '12')
        and a.scheme not in (
            select distinct fm_scheme
            from bronze.canara_dbo_mcr_schexclude_ro
        )
        and concat(a.scheme, a.plan) not in (
            select concat(scheme, plan)
            from bronze.global_plan_exclude
            where fund = p.fund_code
        )
    group by a.scheme,
        a.plan,
        a.fund
),
-- ============================================================
-- 5️⃣  Current Quarter Intra AUM
-- ============================================================
curr_qtr_intra as (
    select d.scheme,
        d.plan,
        d.fund,
        sum(cast(d.aum as decimal(38, 3))) as closeintraaum
    from standardmodel.aum_dview d
        join params p on d.fund = p.fund_code
    where date(d.batchclosedt) = p.curr_qtr_end
        and d.scheme not in (
            select distinct fm_scheme
            from bronze.canara_dbo_mcr_schexclude_ro
        )
        and concat(d.scheme, d.plan) not in (
            select concat(scheme, plan)
            from bronze.global_plan_exclude
            where fund = p.fund_code
        )
    group by d.scheme,
        d.plan,
        d.fund
),
-- ============================================================
-- 6️⃣  Inflow and Outflow
-- ============================================================
flow_data as (
    select i.scheme,
        i.plan,
        i.fund,
        sum(
            case
                when i.trtype in (
                    'NEW',
                    'ADD',
                    'SIN',
                    'IPO',
                    'LTIN',
                    'LTIA',
                    'STPN',
                    'STPA',
                    'STPI'
                ) then cast(i.cramt as decimal(38, 3))
                else 0
            end
        ) as puramt,
        sum(
            case
                when i.trtype in ('RED', 'FUL', 'SWD', 'TRG', 'LTOF', 'LTOP', 'STPO') then cast(i.dbamt as decimal(38, 3))
                else 0
            end
        ) as redamt
    from standardmodel.inout_flow_dview i
        join params p on i.fund = p.fund_code
    where date(i.batchclosedt) between p.prev_qtr_end and p.curr_qtr_end
    group by i.scheme,
        i.plan,
        i.fund
),
-- ============================================================
-- 7️⃣  Folio Count (as of Current Quarter End)
-- ============================================================
folio_data as (
    select f.scheme,
        f.plan,
        f.fund,
        count(distinct f.folio) as folio_count
    from standardmodel.folio_meview f
        join params p on f.fund = p.fund_code
    where date(
            from_iso8601_date(
                concat(f.creation_year, '-', f.creation_month, '-01')
            )
        ) <= p.curr_qtr_end
        and f.scheme not in (
            select distinct fm_scheme
            from bronze.canara_dbo_mcr_schexclude_ro
        )
        and concat(f.scheme, f.plan) not in (
            select concat(scheme, plan)
            from bronze.global_plan_exclude
            where fund = p.fund_code
        )
    group by f.scheme,
        f.plan,
        f.fund
),
-- ============================================================
-- 8️⃣  Combine All Metrics
-- ============================================================
combined as (
    select coalesce(
            a.scheme,
            ia.scheme,
            ca.scheme,
            cia.scheme,
            fl.scheme,
            fd.scheme
        ) as scheme,
        coalesce(
            a.plan,
            ia.plan,
            ca.plan,
            cia.plan,
            fl.plan,
            fd.plan
        ) as plan,
        coalesce(
            a.fund,
            ia.fund,
            ca.fund,
            cia.fund,
            fl.fund,
            fd.fund
        ) as fund,
        coalesce(a.openaum, 0) as openaum,
        coalesce(ia.openintraaum, 0) as openintraaum,
        coalesce(ca.closeaum, 0) as closeaum,
        coalesce(cia.closeintraaum, 0) as closeintraaum,
        coalesce(fd.puramt, 0) as puramt,
        coalesce(fd.redamt, 0) as redamt,
        coalesce(fl.folio_count, 0) as folio_count
    from prev_qtr_aum a
        full outer join prev_qtr_intra ia on a.scheme = ia.scheme
        and a.plan = ia.plan
        and a.fund = ia.fund
        full outer join curr_qtr_aum ca on coalesce(a.scheme, ia.scheme) = ca.scheme
        and coalesce(a.plan, ia.plan) = ca.plan
        full outer join curr_qtr_intra cia on coalesce(ca.scheme, a.scheme) = cia.scheme
        and coalesce(ca.plan, a.plan) = cia.plan
        full outer join flow_data fd on coalesce(a.scheme, ca.scheme) = fd.scheme
        and coalesce(a.plan, ca.plan) = fd.plan
        full outer join folio_data fl on coalesce(a.scheme, ca.scheme) = fl.scheme
        and coalesce(a.plan, ca.plan) = fl.plan
),
-- ============================================================
-- 9️⃣  Scheme Master for Names & Categories
-- ============================================================
scheme_master as (
    select scheme_code,
        scheme_name,
        final_category
    from bronze.global_scheme_master
) -- ============================================================
-- 🔟  Final Output
-- ============================================================
select c.scheme,
    g.scheme_name as "Scheme Name",
    g.final_category as "Scheme Category (As per MCR)",
    round(
        (c.openaum - coalesce(c.openintraaum, 0)) / 1e7,
        2
    ) as "AUM as on last day of previous quarter (INR Cr)",
    c.folio_count as "Folio Count",
    round((c.puramt) / 1e7, 2) as "Total Inflow (INR Cr)",
    round((c.redamt) / 1e7, 2) as "Total Outflow (INR Cr)",
    round(
        (c.closeaum - coalesce(c.closeintraaum, 0)) / 1e7,
        2
    ) as "AUM as on last day of current quarter (INR Cr)"
from combined c
    left join scheme_master g on c.scheme = g.scheme_code
order by g.final_category,
    g.scheme_name;