with params as (
    select date('2025-06-30') as curr_qtr_end,
        '125' as fund_code
),
quarter_logic as (
    select curr_qtr_end,
        case
            when extract(
                month
                from curr_qtr_end
            ) = 6 then null 
            when extract(
                month
                from curr_qtr_end
            ) = 9 then date('2025-06-30') 
            when extract(
                month
                from curr_qtr_end
            ) = 12 then date('2025-09-30')
            when extract(
                month
                from curr_qtr_end
            ) = 3 then date('2025-12-31') 
        end as prev_qtr_end
    from params
),
prev_qtr_aum as (
    select a.scheme,
        a.plan,
        a.fund,
        sum(cast(a.aum as decimal(38, 3))) as openaum
    from standardmodel.aum_meview a
        join params p on a.fund = p.fund_code
        join quarter_logic q on 1 = 1
    where q.prev_qtr_end is not null
        and date(
            from_iso8601_date(
                concat(a.creation_year, '-', a.creation_month, '-01')
            )
        ) <= q.prev_qtr_end
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
prev_qtr_intra as (
    select d.scheme,
        d.plan,
        d.fund,
        sum(cast(d.aum as decimal(38, 3))) as openintraaum
    from standardmodel.aum_dview d
        join params p on d.fund = p.fund_code
        join quarter_logic q on 1 = 1
    where q.prev_qtr_end is not null
        and cast(d.batchclosedt as date) = q.prev_qtr_end
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
prev_qtr_etf as (
    select e.scheme,
        e.plan,
        e.fund,
        sum(cast(e.aum as decimal(38, 3))) as openaum_etf
    from standardmodel.etf_aum_dview e
        join params p on e.fund = p.fund_code
        join quarter_logic q on 1 = 1
    where q.prev_qtr_end is not null
        and cast(e.rundt as date) = q.prev_qtr_end
        and e.scheme not in (
            select distinct fm_scheme
            from bronze.canara_dbo_mcr_schexclude_ro
        )
        and concat(e.scheme, e.plan) not in (
            select concat(scheme, plan)
            from bronze.global_plan_exclude
            where fund = p.fund_code
        )
    group by e.scheme,
        e.plan,
        e.fund
),
curr_qtr_aum as (
    select a.scheme,
        a.plan,
        a.fund,
        sum(cast(a.aum as decimal(38, 3))) as closeaum
    from standardmodel.aum_meview a
        join params p on a.fund = p.fund_code
        join quarter_logic q on 1 = 1
    where date(
            from_iso8601_date(
                concat(a.creation_year, '-', a.creation_month, '-01')
            )
        ) <= q.curr_qtr_end
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
curr_qtr_intra as (
    select d.scheme,
        d.plan,
        d.fund,
        sum(cast(d.aum as decimal(38, 3))) as closeintraaum
    from standardmodel.aum_dview d
        join params p on d.fund = p.fund_code
        join quarter_logic q on 1 = 1
    where cast(d.batchclosedt as date) = q.curr_qtr_end
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
                    'STPI',
                    'NEWR',
                    'ADDR',
                    'SINR',
                    'IPOR',
                    'LTINR',
                    'LTIAR',
                    'STPNR',
                    'STPAR',
                    'STPIR',
                    'DIR',
                    'DIRR',
                    'DSPI',
                    'DSPIR'
                )
                and i.purred = 'P' then cast(i.cramt as decimal(38, 3))
                when i.trtype in (
                    'NEW',
                    'ADD',
                    'SIN',
                    'IPO',
                    'LTIN',
                    'LTIA',
                    'STPN',
                    'STPA',
                    'STPI',
                    'NEWR',
                    'ADDR',
                    'SINR',
                    'IPOR',
                    'LTINR',
                    'LTIAR',
                    'STPNR',
                    'STPAR',
                    'STPIR',
                    'DIR',
                    'DIRR',
                    'DSPI',
                    'DSPIR'
                )
                and i.purred = 'R' then cast(i.dbamt as decimal(38, 3)) * -1
                else 0
            end
        ) as puramt,
        sum(
            case
                when i.trtype in (
                    'RED',
                    'FUL',
                    'SWD',
                    'TRG',
                    'LTOF',
                    'LTOP',
                    'STPO',
                    'REDR',
                    'FULR',
                    'SWDR',
                    'TRGR',
                    'LTOFR',
                    'LTOPR',
                    'STPOR'
                )
                and i.purred = 'P' then cast(i.cramt as decimal(38, 3))
                when i.trtype in (
                    'RED',
                    'FUL',
                    'SWD',
                    'TRG',
                    'LTOF',
                    'LTOP',
                    'STPO',
                    'REDR',
                    'FULR',
                    'SWDR',
                    'TRGR',
                    'LTOFR',
                    'LTOPR',
                    'STPOR'
                )
                and i.purred = 'R' then cast(i.dbamt as decimal(38, 3)) * -1
                else 0
            end
        ) as redamt
    from standardmodel.inout_flow_dview i
        join params p on i.fund = p.fund_code
        join quarter_logic q on 1 = 1
    where cast(i.batchclosedt as date) between q.prev_qtr_end and q.curr_qtr_end
    group by i.scheme,
        i.plan,
        i.fund
),
folio_data as (
    select f.scheme,
        f.plan,
        f.fund,
        count(distinct f.folio) as folio_count
    from standardmodel.folio_meview f
        join params p on f.fund = p.fund_code
        join quarter_logic q on 1 = 1
    where date(
            from_iso8601_date(
                concat(f.creation_year, '-', f.creation_month, '-01')
            )
        ) <= q.curr_qtr_end
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
combined as (
    select coalesce(
            ca.scheme,
            pa.scheme,
            ci.scheme,
            pi.scheme,
            pe.scheme,
            fl.scheme,
            fd.scheme
        ) as scheme,
        coalesce(
            ca.plan,
            pa.plan,
            ci.plan,
            pi.plan,
            pe.plan,
            fl.plan,
            fd.plan
        ) as plan,
        coalesce(
            ca.fund,
            pa.fund,
            ci.fund,
            pi.fund,
            pe.fund,
            fl.fund,
            fd.fund
        ) as fund,
        coalesce(pa.openaum, 0) + coalesce(pe.openaum_etf, 0) as openaum,
        coalesce(pi.openintraaum, 0) as openintraaum,
        coalesce(ca.closeaum, 0) as closeaum,
        coalesce(ci.closeintraaum, 0) as closeintraaum,
        coalesce(fd.puramt, 0) as puramt,
        coalesce(fd.redamt, 0) as redamt,
        coalesce(fl.folio_count, 0) as folio_count
    from curr_qtr_aum ca
        full outer join curr_qtr_intra ci on ca.scheme = ci.scheme
        and ca.plan = ci.plan
        full outer join prev_qtr_aum pa on coalesce(ca.scheme, ci.scheme) = pa.scheme
        and coalesce(ca.plan, ci.plan) = pa.plan
        full outer join prev_qtr_intra pi on coalesce(pa.scheme) = pi.scheme
        and coalesce(pa.plan) = pi.plan
        full outer join prev_qtr_etf pe on coalesce(pa.scheme, pi.scheme) = pe.scheme
        and coalesce(pa.plan, pi.plan) = pe.plan
        full outer join folio_data fl on coalesce(ca.scheme, pa.scheme) = fl.scheme
        and coalesce(ca.plan, pa.plan) = fl.plan
        full outer join flow_data fd on coalesce(ca.scheme, pa.scheme) = fd.scheme
        and coalesce(ca.plan, pa.plan) = fd.plan
),
scheme_master as (
    select scheme_code,
        scheme_name,
        final_category
    from bronze.global_scheme_master
)
select c.scheme,
    g.scheme_name as "Scheme Name",
    g.final_category as "Scheme Category (As per MCR)",
    case
        when extract(
            month
            from q.curr_qtr_end
        ) = 6 then 0
        else round(
            (c.openaum - coalesce(c.openintraaum, 0)) / 1e7,
            2
        )
    end as "AUM as on last day of previous quarter (INR Cr)",
    c.folio_count as "Folio Count",
    round((c.puramt) / 1e7, 2) as "Total Inflow (INR Cr)",
    round((c.redamt) / 1e7, 2) as "Total Outflow (INR Cr)",
    round(
        (c.closeaum - coalesce(c.closeintraaum, 0)) / 1e7,
        2
    ) as "AUM as on last day of current quarter (INR Cr)"
from combined c
    left join scheme_master g on c.scheme = g.scheme_code
    cross join quarter_logic q
order by g.final_category,
    g.scheme_name;