query_mlb = """WITH NPS AS(
            SELECT RES.*,
            CASE WHEN EXTRACT(MONTH FROM cast(RES.END_DATE as date)) BETWEEN 1 AND 3 THEN 'Q1'
                 WHEN EXTRACT(MONTH FROM cast(RES.END_DATE as date)) BETWEEN 4 AND 6 THEN 'Q2'
                 WHEN EXTRACT(MONTH FROM cast(RES.END_DATE as date)) BETWEEN 7 AND 9 THEN 'Q3'
                 WHEN EXTRACT(MONTH FROM cast(RES.END_DATE as date)) BETWEEN 10 AND 12 THEN 'Q4' END AS QUARTER,
            EXTRACT(YEAR FROM cast(RES.END_DATE as date)) AS YEAR,
            COALESCE(PRED.ORD_ORDER_ID, -1) AS ORDER_DUP
             FROM `meli-bi-data.SBOX_NPS_ANALYTICS.NPS_TX_MERGE` RES
             LEFT JOIN `meli-bi-data.SBOX_CX_BI_ADS_CORE.LK_CX_NPS_TX_PREDICTIONS` PRED
             ON RES.ORDER_ID = PRED.ORD_ORDER_ID
             
            WHERE
                RES.ROL in ('B')
                -- AND RES.SHIPPING_SELLER in ('drop off','xd','fulfillment','ME Flex')
                and RES.SITE in ('MLB')
                and RES.NPS = -1
                and cast(RES.END_DATE as date) = {date}
        )

        SELECT 
            SITE,
            BUYER_ID as CUST_ID,
            ORDER_ID,
            CAST(END_DATE as DATE) AS END_DATE,
            NOTA_NPS,
            NPS.NPS,
            DETRACTION_REASON_NPS AS DETRACTION_REASON,
            COMMENTS,
            SURVEY_ID
        FROM NPS
        WHERE ORDER_DUP = -1
        """