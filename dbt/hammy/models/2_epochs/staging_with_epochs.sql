{{
    config(
        materialized='view', 
    )
}}

WITH rn AS
(SELECT *, toInt32(ROW_NUMBER() OVER(ORDER BY tag, experimentNumber, hypothesisNumber, implementationNumber,
  programTimestamp, threadNumber, chunkNumber, epoch)) AS epochId FROM {{ ref('epochs') }})

SELECT *
FROM {{ ref('staging') }} JOIN rn
USING (tag, experimentNumber, hypothesisNumber, implementationNumber,
  programTimestamp, threadNumber, chunkNumber, epoch)
