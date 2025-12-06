select c_count, count(*) as custdist
from (
    select
        c_custkey,
        count(o_orderkey) as c_count
    from
        customer left outer join orders on
            c_custkey = o_custkey
            and o_comment not like '%special%requests%'
    group by  
        c_custkey
    ) as c_orders
group by
    c_count
order by
    custdist desc,
    c_count desc;

--paths = [
--    "s3://activeloopai-db-dev--use1-az6--x-s3/customer/",
--    "s3://activeloopai-db-dev--use1-az6--x-s3/lineiten/",
--    "s3://activeloopai-db-dev--use1-az6--x-s3/nation/",
--    "s3://activeloopai-db-dev--use1-az6--x-s3/orders/",
--    "s3://activeloopai-db-dev--use1-az6--x-s3/part/",
--    "s3://activeloopai-db-dev--use1-az6--x-s3/partsupp/",
--    "s3://activeloopai-db-dev--use1-az6--x-s3/region/",
--    "s3://activeloopai-db-dev--use1-az6--x-s3/supplier/",
--]