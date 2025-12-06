DROP TABLE IF EXISTS customer;
DROP TABLE IF EXISTS lineitem;
DROP TABLE IF EXISTS nation;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS part;
DROP TABLE IF EXISTS partsupp;
DROP TABLE IF EXISTS region;
DROP TABLE IF EXISTS supplier;

CREATE TABLE customer (
    c_custkey int NOT NULL,
    c_name VARCHAR(25) NOT NULL,
    c_address VARCHAR(40) NOT NULL,
    c_nationkey int NOT NULL,
    c_phone VARCHAR(15) NOT NULL,
    c_acctbal decimal(15, 2) NOT NULL,
    c_mktsegment VARCHAR(10) NOT NULL,
    c_comment VARCHAR(117) NOT NULL
) USING deeplake;

CREATE TABLE lineitem (
    l_orderkey int NOT NULL,
    l_partkey int NOT NULL,
    l_suppkey int not null,
    l_linenumber int not null,
    l_quantity decimal(15, 2) NOT NULL,
    l_extendedprice decimal(15, 2) NOT NULL,
    l_discount decimal(15, 2) NOT NULL,
    l_tax decimal(15, 2) NOT NULL,
    l_returnflag VARCHAR(1) NOT NULL,
    l_linestatus VARCHAR(1) NOT NULL,
    l_shipdate DATE NOT NULL,
    l_commitdate DATE NOT NULL,
    l_receiptdate DATE NOT NULL,
    l_shipinstruct VARCHAR(25) NOT NULL,
    l_shipmode VARCHAR(10) NOT NULL,
    l_comment VARCHAR(44) NOT NULL
) USING deeplake;

CREATE TABLE nation (
  n_nationkey int NOT NULL,
  n_name varchar(25) NOT NULL,
  n_regionkey int NOT NULL,
  n_comment varchar(152) NULL
) USING deeplake;

CREATE TABLE orders (
    o_orderkey int NOT NULL,
    o_custkey int NOT NULL,
    o_orderstatus VARCHAR(1) NOT NULL,
    o_totalprice decimal(15, 2) NOT NULL,
    o_orderdate DATE NOT NULL,
    o_orderpriority VARCHAR(15) NOT NULL,
    o_clerk VARCHAR(15) NOT NULL,
    o_shippriority int NOT NULL,
    o_comment VARCHAR(79) NOT NULL
) USING deeplake;

CREATE TABLE part (
    p_partkey int NOT NULL,
    p_name VARCHAR(55) NOT NULL,
    p_mfgr VARCHAR(25) NOT NULL,
    p_brand VARCHAR(10) NOT NULL,
    p_type VARCHAR(25) NOT NULL,
    p_size int NOT NULL,
    p_container VARCHAR(10) NOT NULL,
    p_retailprice decimal(15, 2) NOT NULL,
    p_comment VARCHAR(23) NOT NULL
) USING deeplake;

CREATE TABLE partsupp (
    ps_partkey int NOT NULL,
    ps_suppkey int NOT NULL,
    ps_availqty int NOT NULL,
    ps_supplycost decimal(15, 2) NOT NULL,
    ps_comment VARCHAR(199) NOT NULL
) USING deeplake;

CREATE TABLE region (
    r_regionkey int NOT NULL,
    r_name VARCHAR(25) NOT NULL,
    r_comment VARCHAR(152)
) USING deeplake;

CREATE TABLE supplier (  
    s_suppkey int NOT NULL,
    s_name VARCHAR(25) NOT NULL,
    s_address VARCHAR(40) NOT NULL,
    s_nationkey int NOT NULL,
    s_phone VARCHAR(15) NOT NULL,
    s_acctbal decimal(15, 2) NOT NULL,
    s_comment VARCHAR(101) NOT NULL
) USING deeplake;

create view revenue0 (supplier_no, total_revenue) as
select
    l_suppkey,
    sum(l_extendedprice * (1 - l_discount))
from
    lineitem
where
    l_shipdate >= date '1996-01-01'
    and l_shipdate < date '1996-01-01' + interval '3' month
group by
    l_suppkey;
