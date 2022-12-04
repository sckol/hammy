{{
    config(
        materialized='incremental', 
        order_by=('epochId', 'time')
    )
}}

SELECT epochId, time, position FROM {{ ref('staging_with_epochs') }}
WHERE 
tag = 'walk' AND experimentNumber = 1 AND hypothesisNumber = 1
AND implementationNumber = 1
{% if is_incremental() %}
AND programTimestamp > (SELECT MAX(programTimestamp) FROM {{ this }})
{% endif %} 
