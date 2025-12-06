select
  s_suppkey,
  s_name,
  s_address,
  s_phone,
  total_revenue
from
  supplier,
  revenue0
where
  s_suppkey = supplier_no
  and abs(total_revenue - (
    select
      max(total_revenue)
    from
      revenue0
  )) < 1e-6
order by
  s_suppkey;
