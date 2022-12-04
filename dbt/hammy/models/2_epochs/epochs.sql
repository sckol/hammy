{{
    config(
        materialized='incremental', 
        order_by=('tag', 'experimentNumber', 'hypothesisNumber', 'implementationNumber',
          'programTimestamp', 'threadNumber', 'chunkNumber', 'epoch')
    )
}}

SELECT DISTINCT tag, experimentNumber, hypothesisNumber, implementationNumber,
  programTimestamp, threadNumber, chunkNumber, epoch
FROM {{ ref('staging') }} 
{% if is_incremental() %}
WHERE tuple(tag, experimentNumber, hypothesisNumber, implementationNumber,
  programTimestamp, threadNumber, chunkNumber, epoch) NOT IN
  (SELECT * FROM {{ this }})
{% endif %}  