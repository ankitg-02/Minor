WITH params AS (
  SELECT
    '07' AS report_month,
    '2025' AS report_year
),
dates AS (
  SELECT
    report_month,
    report_year,
    CASE
      WHEN report_month = '04' THEN NULL -- Q1 => prev quarter = 0
      WHEN report_month = '07' THEN concat(report_year, '-03-31')
      WHEN report_month = '10' THEN concat(report_year, '-06-30')
      WHEN report_month = '01' THEN concat(cast(cast(report_year AS integer) - 1 AS varchar), '-09-30')
      ELSE NULL
    END AS prev_quarter_end_str,
    CASE
      WHEN report_month = '04' THEN concat(report_year, '-06-30')
      WHEN report_month = '07' THEN concat(report_year, '-09-30')
      WHEN report_month = '10' THEN concat(report_year, '-12-31')
      WHEN report_month = '01' THEN concat(report_year, '-03-31')
      ELSE NULL
    END AS curr_quarter_end_str
  FROM params
),

dates_parsed AS (
  SELECT
    report_month,
    report_year,
    date(prev_quarter_end_str) AS prev_quarter_end_date,
    date(curr_quarter_end_str) AS curr_quarter_end_date
  FROM dates
),

-- ===========================================================
-- t1: main AUM aggregation for mutual + ETF (your original logic)
-- ===========================================================
t1 AS (
  select *
  from (
    select scheme,
      sum(folio_count) folio_count,
      sum(openaum) openaum,
      sum(puramt) puramt,
      sum(redamt) redamt,
      sum(closeaum) closeaum
    from (
      select coalesce(a.scheme, b.scheme, c.scheme) as scheme,
        coalesce(b.folio_count, 0) as folio_count,
        coalesce(a.openaum, 0) as openaum,
        coalesce(c.puramt, 0) as puramt,
        coalesce(c.redamt, 0) as redamt,
        coalesce(b.closeaum, 0) as closeaum
      from (
        select scheme, sum(aum) as openaum
        from standardmodel.aum_meview a
        where fund = '125'
          and scheme not in (
            select fm_scheme from bronze.indiabulls_dbo_mcr_schexclude_rt
          )
          and concat(scheme, plan) not in (
            select concat(scheme, plan)
            from bronze.global_plan_exclude
            where fund = '125'
          )
          and creation_month = '04'  -- current quarter start (dynamic later)
          and creation_year = '2025'
        group by 1
      ) as a
      full join (
        select scheme,
          count(distinct (scheme, plan, folio)) folio_count,
          sum(aum) closeaum
        from (
          select scheme, plan, folio, sum(aum) aum
          from standardmodel.aum_meview
          where fund = '125'
            and scheme not in (
              select fm_scheme from bronze.indiabulls_dbo_mcr_schexclude_rt
            )
            and concat(scheme, plan) not in (
              select concat(scheme, plan)
              from bronze.global_plan_exclude_table
              where fund = '125'
            )
            and creation_month = '06'
            and creation_year = '2025'
          group by 1,2,3
          having sum(netunits) > 0
        )
        group by 1
      ) as b on a.scheme = b.scheme
      full join (
        select scheme,
          sum(
            case
              when trtype in ('NEW','ADD','SIN','IPO','LTIN','LTIA','STPN','STPA','STPI','NEWR','ADDR','SINR','IPOR','LTINR','LTIAR','STPNR','STPAR','STPIR','DIR','DIRR','DSPI','DSPIR')
              and purred = 'P' then cramt
              when trtype in ('NEW','ADD','SIN','IPO','LTIN','LTIA','STPN','STPA','STPI','NEWR','ADDR','SINR','IPOR','LTINR','LTIAR','STPNR','STPAR','STPIR','DIR','DIRR','DSPI','DSPIR')
              and purred = 'R' then dbamt * -1
              else 0
            end
          ) as puramt,
          sum(
            case
              when trtype in ('RED','FUL','SWD','TRG','LTOF','LTOP','STPO','REDR','FULR','SWDR','TRGR','LTOFR','LTOPR','STPOR')
              and purred = 'P' then -(cramt + exitload)
              when trtype in ('RED','FUL','SWD','TRG','LTOF','LTOP','STPO','REDR','FULR','SWDR','TRGR','LTOFR','LTOPR','STPOR')
              and purred = 'R' then dbamt + exitload
              else 0
            end
          ) as redamt
        from standardmodel.inout_flow_dview
        where fund = '125'
          and batchclosedt between '2025-04-01' and '2025-06-30'
        group by 1
      ) as c on c.scheme = a.scheme
    )
    group by 1
  )
  union
  select *
  from (
    select scheme,
      sum(folio_count) folio_count,
      sum(openaum) openaum,
      sum(puramt) puramt,
      sum(redamt) redamt,
      sum(closeaum) closeaum
    from (
      select coalesce(a.scheme, b.scheme, c.scheme) scheme,
        coalesce(b.folio_count, 0) as folio_count,
        coalesce(a.openaum, 0) openaum,
        coalesce(c.puramt, 0) puramt,
        coalesce(c.redamt, 0) redamt,
        coalesce(b.closeaum, 0) closeaum
      from (
        select scheme, sum(cast(aum as decimal(38,3))) as openaum
        from standardmodel.etf_aum_dview
        where fund = '125' and rundt = '2025-03-31'
        group by 1
      ) as a
      full join (
        select scheme, count(distinct(scheme, plan, dpclid)) folio_count,
          sum(cast(aum as decimal(38,3))) as closeaum
        from (
          select scheme, plan, dpclid, sum(cast(aum as decimal(38,3))) as aum
          from standardmodel.etf_aum_dview
          where fund = '125' and rundt = '2025-06-30'
          group by 1,2,3
          having sum(netunits) > 0
        )
        group by 1
      ) as b on a.scheme = b.scheme
      full join (
        select scheme,
          sum(case when purred = 'P' then apln_amount else 0 end) as puramt,
          sum(case when purred = 'R' then apln_amount else 0 end) as redamt
        from standardmodel.etf_inout_flow_dview
        where fund = '125'
          and rundt between '2025-04-01' and '2025-06-30'
        group by 1
      ) as c on a.scheme = c.scheme
    )
    group by 1
  )
),

-- ===========================================================
-- PREVIOUS QUARTER AUM & INTRA AUM (dynamic FY logic)
-- ===========================================================
prev_openaum AS (
  SELECT a.scheme, sum(aum) AS prev_openaum
  FROM standardmodel.aum_meview a
  JOIN dates_parsed d ON TRUE
  WHERE d.prev_quarter_end_date IS NOT NULL
    AND date(concat(a.creation_year, '-', lpad(a.creation_month,2,'0'), '-01')) <= d.prev_quarter_end_date
  GROUP BY 1
),
prev_open_intra AS (
  SELECT
    g.scheme,
    sum(a.aum) AS prev_openintraaum
  FROM standardmodel.aum_dview a
  JOIN bronze.global_intra_scheme g
    ON g.folio = a.folio AND g.fund = a.fund
  JOIN dates_parsed d ON TRUE
  WHERE d.prev_quarter_end_date IS NOT NULL
    AND date(a.batchclosedt) = d.prev_quarter_end_date
  GROUP BY 1
),

x1 AS (
  SELECT
    g.scheme,
    sum(a.aum) AS closeintraaum
  FROM standardmodel.aum_dview a
  JOIN bronze.global_intra_scheme g
    ON g.folio = a.folio AND g.fund = a.fund
  JOIN dates_parsed d ON TRUE
  WHERE date(a.batchclosedt) = d.curr_quarter_end_date
  GROUP BY 1
),

liquid AS (
  select trs_scheme as scheme,
    sum(
      case
        when trs_purred = 'P' then coalesce (cast(trs_cramt as decimal(38, 2)), 0)
        when trs_purred = 'R' then coalesce(cast(trs_dbamt as decimal(38, 2))) * -1
      end
    ) as liquidaum
  from bronze.indiabulls_dbo_mtrans_file_rt
  where trs_trtype in ('NEW','ADD','SIN','IPO','NEWR','ADDR','SINR','IPOR')
  group by 1
)

-- ===========================================================
-- FINAL OUTPUT WITH RECTIFIED PREVIOUS QUARTER LOGIC
-- ===========================================================
SELECT
  y.scheme,
  g.scheme_name AS "Scheme Name",
  g.final_category AS "Scheme Category (As per MCR)",
  CASE
    WHEN p.report_month = '04' THEN 0
    ELSE (coalesce(po.prev_openaum, 0) - coalesce(pi.prev_openintraaum, 0)) / 1e7
  END AS "AUM as on last day of previous quarter (INR Cr)",
  y.folio_count AS "Folio Count",
  (y.puramt - coalesce(l.liquidaum, 0)) / 1e7 AS "Total Inflow (INR Cr)",
  (y.redamt) / 1e7 AS "Total Outflow (INR Cr)",
  (y.closeaum - coalesce(x1.closeintraaum, 0)) / 1e7 AS "AUM as on last day of current quarter (INR Cr)"
FROM t1 y
CROSS JOIN params p
LEFT JOIN prev_openaum po ON po.scheme = y.scheme
LEFT JOIN prev_open_intra pi ON pi.scheme = y.scheme
LEFT JOIN x1 ON x1.scheme = y.scheme
LEFT JOIN liquid l ON l.scheme = y.scheme