--------------------------------------------------------------------------------
-- Full query for AWS Athena (Presto dialect)
-- Purpose: Correctly compute "AUM as on last day of previous quarter (INR Cr)"
--          with Fiscal Year Apr-Mar; Q1 (Apr-Jun) previous AUM forced to 0.
-- Note: change params.report_end_dt to the last day of the quarter you are running.
--------------------------------------------------------------------------------
WITH
-- ====== PARAMETERS ======
params AS (
  -- For Quarter 1 (Apr-Jun) example set: 2025-06-30
  SELECT DATE '2025-06-30' AS report_end_dt
),

-- ====== QUARTER / FY LOGIC ======
q AS (
  SELECT
    report_end_dt,
    extract(month FROM report_end_dt) AS month_num,
    -- Convert month to fiscal month where Apr = 1, May = 2, ..., Mar = 12
    CASE
      WHEN extract(month FROM report_end_dt) >= 4 THEN extract(month FROM report_end_dt) - 3
      ELSE extract(month FROM report_end_dt) + 9
    END AS fiscal_month,
    -- Fiscal quarter 1..4
    CAST(floor((CASE WHEN extract(month FROM report_end_dt) >= 4 THEN extract(month FROM report_end_dt) - 3 ELSE extract(month FROM report_end_dt) + 9 END - 1) / 3) + 1 AS INTEGER) AS fiscal_quarter,
    -- Last day of previous quarter: date_trunc('quarter', report_end_dt) - 1 day
    CAST(date_add('day', -1, date_trunc('quarter', report_end_dt)) AS date) AS prev_quarter_end_dt,
    -- First day of current quarter (useful for flow windows)
    CAST(date_trunc('quarter', report_end_dt) AS date) AS curr_quarter_start_dt
  FROM params
),

-- ========== PREVIOUS QUARTER: OPEN AUM (non-ETF) ==========
prev_open_non_etf AS (
  -- Only fetch when fiscal_quarter > 1; for Q1 this CTE yields zero rows -> treated as 0 downstream
  SELECT
    scheme,
    SUM(aum) AS openaum
  FROM standardmodel.aum_meview
  CROSS JOIN q
  WHERE fund = '125'
    AND q.fiscal_quarter > 1
    -- use prev_quarter_end_dt as the creation/measurement date for previous-quarter open aum
    AND CAST(creation_date AS date) = q.prev_quarter_end_dt
    AND scheme NOT IN (SELECT distinct fm_scheme FROM bronze.indiabulls_dbo_mcr_schexclude_rt)
    AND concat(scheme, plan) NOT IN (SELECT concat(scheme, plan) FROM bronze.global_plan_exclude_table WHERE fund = '125')
  GROUP BY scheme
),

-- ========== PREVIOUS QUARTER: OPEN AUM (ETF) ==========
prev_open_etf AS (
  SELECT
    scheme,
    SUM(CAST(aum AS double)) AS openaum
  FROM standardmodel.etf_aum_dview
  CROSS JOIN q
  WHERE fund = '125'
    AND q.fiscal_quarter > 1
    AND CAST(rundt AS date) = q.prev_quarter_end_dt
  GROUP BY scheme
),

-- ========== PREVIOUS QUARTER: INTRA OPEN (folios that are intra/DPCL) non-ETF ==========
prev_open_intra_non_etf AS (
  -- This uses mapping table bronze.global_intra_scheme to map folio -> scheme for intra folios
  SELECT
    gis.scheme AS scheme,
    SUM(a.netunits) AS openintraunits,
    SUM(a.aum) AS openintraaum
  FROM standardmodel.aum_dview a
  JOIN bronze.global_intra_scheme gis
    ON gis.fund = a.fund AND gis.folio = a.folio
  CROSS JOIN q
  WHERE a.fund = '125'
    AND q.fiscal_quarter > 1
    AND CAST(a.batchclosedt AS date) = q.prev_quarter_end_dt
  GROUP BY gis.scheme
),

-- ========== PREVIOUS QUARTER: INTRA OPEN (ETF) ==========
prev_open_intra_etf AS (
  SELECT
    gis.scheme AS scheme,
    SUM(CAST(a.netunits AS double)) AS openintraunits,
    SUM(CAST(a.aum AS double)) AS openintraaum
  FROM standardmodel.etf_aum_dview a
  JOIN bronze.global_intra_scheme gis
    ON gis.fund = a.fund AND gis.folio = a.dpclid  -- dpclid used as folio-like identifier in ETF view
  CROSS JOIN q
  WHERE a.fund = '125'
    AND q.fiscal_quarter > 1
    AND CAST(a.rundt AS date) = q.prev_quarter_end_dt
  GROUP BY gis.scheme
),

-- ========== CURRENT QUARTER: CLOSE AUM (non-ETF) ==========
curr_close_non_etf AS (
  SELECT
    scheme,
    COUNT(DISTINCT concat(scheme, '-', plan, '-', folio)) AS folio_count,
    SUM(aum) AS closeaum
  FROM (
    SELECT scheme, plan, folio, SUM(aum) AS aum, SUM(netunits) AS netunits
    FROM standardmodel.aum_meview
    WHERE fund = '125'
      AND scheme NOT IN (SELECT distinct fm_scheme FROM bronze.indiabulls_dbo_mcr_schexclude_rt)
      AND concat(scheme, plan) NOT IN (SELECT concat(scheme, plan) FROM bronze.global_plan_exclude_table WHERE fund = '125')
      -- current quarter last day: use params.report_end_dt
      AND CAST(creation_date AS date) = (SELECT report_end_dt FROM params)
    GROUP BY scheme, plan, folio
    HAVING SUM(netunits) > 0
  )
  GROUP BY scheme
),

-- ========== CURRENT QUARTER: CLOSE AUM (ETF) ==========
curr_close_etf AS (
  SELECT
    scheme,
    COUNT(DISTINCT concat(scheme, '-', dpclid)) AS folio_count,
    SUM(CAST(aum AS double)) AS closeaum
  FROM standardmodel.etf_aum_dview
  WHERE fund = '125'
    AND CAST(rundt AS date) = (SELECT report_end_dt FROM params)
  GROUP BY scheme
),

-- ========== CURRENT QUARTER: INTRA CLOSE (non-ETF) ==========
curr_close_intra_non_etf AS (
  SELECT
    gis.scheme AS scheme,
    SUM(a.netunits) AS closeintraunits,
    SUM(a.aum) AS closeintraaum
  FROM standardmodel.aum_dview a
  JOIN bronze.global_intra_scheme gis
    ON gis.fund = a.fund AND gis.folio = a.folio
  WHERE a.fund = '125'
    AND CAST(a.batchclosedt AS date) = (SELECT report_end_dt FROM params)
  GROUP BY gis.scheme
),

-- ========== CURRENT QUARTER: INTRA CLOSE (ETF) ==========
curr_close_intra_etf AS (
  SELECT
    gis.scheme AS scheme,
    SUM(CAST(a.netunits AS double)) AS closeintraunits,
    SUM(CAST(a.aum AS double)) AS closeintraaum
  FROM standardmodel.etf_aum_dview a
  JOIN bronze.global_intra_scheme gis
    ON gis.fund = a.fund AND gis.folio = a.dpclid
  WHERE a.fund = '125'
    AND CAST(a.rundt AS date) = (SELECT report_end_dt FROM params)
  GROUP BY gis.scheme
),

-- ========== CURRENT QUARTER: INFLOWS / OUTFLOWS (use current quarter window) ==========
curr_inout AS (
  SELECT
    scheme,
    SUM(CASE WHEN trtype IN (
         'NEW','ADD','SIN','IPO','LTIN','LTIA','STPN','STPA','STPI','NEWR','REINV'
        ) THEN coalesce(cramt,0)
        ELSE 0 END) AS total_inflow_amt,
    SUM(CASE WHEN trtype IN (
         'REDEM','SWP','STP','RED' -- example outflow/trtype; adjust to your exact list if different
        ) THEN coalesce(redamt,0)
        ELSE 0 END) AS total_outflow_amt
  FROM standardmodel.inout_flow_dview
  WHERE fund = '125'
    -- flows that happened within current quarter (inclusive)
    AND CAST(batchclosedt AS date) BETWEEN (SELECT curr_quarter_start_dt FROM q) AND (SELECT report_end_dt FROM params)
    AND scheme NOT IN (SELECT distinct fm_scheme FROM bronze.indiabulls_dbo_mcr_schexclude_rt)
    AND concat(scheme, plan) NOT IN (SELECT concat(scheme, plan) FROM bronze.global_plan_exclude WHERE fund = '125')
  GROUP BY scheme
),

-- ========== COMBINE NON-ETF BLOCK ==========
non_etf_block AS (
  SELECT
    coalesce(p.scheme, c.scheme, b.scheme) AS scheme,
    coalesce(b.folio_count, 0) AS folio_count,
    coalesce(p.openaum, 0) AS openaum,
    coalesce(pi.openintraaum, 0) AS openintraaum,
    coalesce(ci.closeintraaum, 0) AS closeintraaum,
    coalesce(b.closeaum, 0) AS closeaum,
    coalesce(ci.closeintraunits, 0) AS closeintraunits,
    coalesce(pi.openintraunits, 0) AS openintraunits,
    coalesce(c.total_inflow_amt, 0) AS puramt,
    coalesce(c.total_outflow_amt, 0) AS redamt
  FROM prev_open_non_etf p
  FULL OUTER JOIN curr_close_non_etf b ON p.scheme = b.scheme
  FULL OUTER JOIN curr_inout c ON coalesce(b.scheme, p.scheme) = c.scheme
  LEFT JOIN prev_open_intra_non_etf pi ON coalesce(b.scheme, p.scheme) = pi.scheme
  LEFT JOIN curr_close_intra_non_etf ci ON coalesce(b.scheme, p.scheme) = ci.scheme
),

-- ========== COMBINE ETF BLOCK ==========
etf_block AS (
  SELECT
    coalesce(p.scheme, c.scheme, b.scheme) AS scheme,
    coalesce(b.folio_count, 0) AS folio_count,
    coalesce(p.openaum, 0) AS openaum,
    coalesce(pi.openintraaum, 0) AS openintraaum,
    coalesce(ci.closeintraaum, 0) AS closeintraaum,
    coalesce(b.closeaum, 0) AS closeaum,
    coalesce(ci.closeintraunits, 0) AS closeintraunits,
    coalesce(pi.openintraunits, 0) AS openintraunits,
    -- Note: ETF flows may be captured separately; reuse curr_inout where ETF entries exist in inout_flow_dview
    coalesce(c.total_inflow_amt, 0) AS puramt,
    coalesce(c.total_outflow_amt, 0) AS redamt
  FROM prev_open_etf p
  FULL OUTER JOIN curr_close_etf b ON p.scheme = b.scheme
  FULL OUTER JOIN curr_inout c ON coalesce(b.scheme, p.scheme) = c.scheme
  LEFT JOIN prev_open_intra_etf pi ON coalesce(b.scheme, p.scheme) = pi.scheme
  LEFT JOIN curr_close_intra_etf ci ON coalesce(b.scheme, p.scheme) = ci.scheme
),

-- ========== UNION: ALL SCHEMES ==========
all_schemes AS (
  SELECT * FROM non_etf_block
  UNION ALL
  SELECT * FROM etf_block
),
final_prep AS (
  SELECT
    a.scheme,
    am.scheme_name,
    am.final_category,
    SUM(a.folio_count) AS folio_count,
    SUM(a.openaum) AS openaum,
    SUM(a.openintraaum) AS openintraaum,
    SUM(a.puramt) AS puramt,
    SUM(a.redamt) AS redamt,
    SUM(a.closeaum) AS closeaum,
    SUM(a.closeintraaum) AS closeintraaum
  FROM all_schemes a
  LEFT JOIN bronze.global_asset_master am
    ON am.fund_code = '125' AND am.scheme_code = a.scheme
  GROUP BY a.scheme, am.scheme_name, am.final_category
)
SELECT
  fp.scheme AS "Scheme Code",
  coalesce(fp.scheme_name, fp.scheme) AS "Scheme Name",
  coalesce(fp.final_category, 'Unknown') AS "Scheme Category (As per MCR)",
  (coalesce(fp.openaum, 0) - coalesce(fp.openintraaum, 0)) / 1e7 AS "AUM as on last day of previous quarter (INR Cr)",
  fp.folio_count AS "Folio Count (Current Quarter End)",
  (coalesce(fp.puramt, 0) - 0) / 1e7 AS "Total Inflow (INR Cr)",  -- adjust subtracting of liquidaum if needed
  (coalesce(fp.redamt, 0)) / 1e7 AS "Total Outflow (INR Cr)",
  (coalesce(fp.closeaum, 0) - coalesce(fp.closeintraaum, 0)) / 1e7 AS "AUM as on last day of current quarter (INR Cr)"
FROM final_prep fp
ORDER BY fp.scheme;
