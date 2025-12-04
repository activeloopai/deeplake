-- INSERT INTO nation
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (0, 'ALGERIA', 0, 'haggle. carefully final deposits detect slyly agai');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (1, 'ARGENTINA', 1, 'al foxes promise slyly according to the regular accounts. bold requests alon');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (2, 'BRAZIL', 1, 'y alongside of the pending deposits. carefully special packages are about the ironic forges. slyly special');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (3, 'CANADA', 1, 'eas hang ironic, silent packages. slyly regular packages are furiously over the tithes. fluffily bold');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (4, 'EGYPT', 4, 'y above the carefully unusual theodolites. final dugouts are quickly across the furiously regular d');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (5, 'ETHIOPIA', 0, 'ven packages wake quickly. regu');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (6, 'FRANCE', 3, 'refully final requests. regular, ironi');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (7, 'GERMANY', 3, 'l platelets. regular accounts x-ray: unusual, regular acco');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (8, 'INDIA', 2, 'ss excuses cajole slyly across the packages. deposits print aroun');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (9, 'INDONESIA', 2, 'slyly express asymptotes. regular deposits haggle slyly. carefully ironic hockey players sleep blithely. carefull');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (10, 'IRAN', 4, 'efully alongside of the slyly final dependencies.');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (11, 'IRAQ', 4, 'nic deposits boost atop the quickly final requests? quickly regula');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (12, 'JAPAN', 2, 'ously. final, express gifts cajole a');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (13, 'JORDAN', 4, 'ic deposits are blithely about the carefully regular pa');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (14, 'KENYA', 0, 'pending excuses haggle furiously deposits. pending, express pinto beans wake fluffily past t');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (15, 'MOROCCO', 0, 'rns. blithely bold courts among the closely regular packages use furiously bold platelets?');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (16, 'MOZAMBIQUE', 0, 's. ironic, unusual asymptotes wake blithely r');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (17, 'PERU', 1, 'platelets. blithely pending dependencies use fluffily across the even pinto beans. carefully silent accoun');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (18, 'CHINA', 2, 'c dependencies. furiously express notornis sleep slyly regular accounts. ideas sleep. depos');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (19, 'ROMANIA', 3, 'ular asymptotes are about the furious multipliers. express dependencies nag above the ironically ironic account');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (20, 'SAUDI ARABIA', 4, 'ts. silent requests haggle. closely express packages sleep across the blithely');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (21, 'VIETNAM', 2, 'hely enticingly express accounts. even, final');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (22, 'RUSSIA', 3, 'requests against the platelets use never according to the quickly regular pint');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (23, 'UNITED KINGDOM', 3, 'eans boost carefully special requests. accounts are. carefull');
INSERT INTO nation (n_nationkey, n_name, n_regionkey, n_comment) VALUES (24, 'UNITED STATES ', 1, 'y final packages. slow foxes cajole quickly. quickly silent platelets breach ironic accounts. unusual pinto be');

-- INSERT INTO region
INSERT INTO region (r_regionkey, r_name, r_comment) VALUES (0, 'AFRICA', 'lar deposits. blithely final packages cajole. regular waters are final requests. regular accounts are according to');
INSERT INTO region (r_regionkey, r_name, r_comment) VALUES (1, 'AMERICA', 'hs use ironic, even requests. s');
INSERT INTO region (r_regionkey, r_name, r_comment) VALUES (2, 'ASIA', 'ges. thinly even pinto beans ca');
INSERT INTO region (r_regionkey, r_name, r_comment) VALUES (3, 'EUROPE', 'ly final courts cajole furiously final excuse');
INSERT INTO region (r_regionkey, r_name, r_comment) VALUES (4, 'MIDDLE EAST', 'uickly special accounts cajole carefully blithely close requests. carefully final asymptotes haggle furiousl');


DO $$
BEGIN
    RAISE NOTICE 'Inserting customer...';
END
$$;
\i sql/tpch/customer.sql
DO $$
BEGIN
    RAISE NOTICE 'Inserting lineitem...';
END
$$;
\i sql/tpch/lineitem_1.sql
\i sql/tpch/lineitem_2.sql
DO $$
BEGIN
    RAISE NOTICE 'Inserting orders...';
END
$$;
\i sql/tpch/orders.sql
DO $$
BEGIN
    RAISE NOTICE 'Inserting part...';
END
$$;
\i sql/tpch/part.sql
DO $$
BEGIN
    RAISE NOTICE 'Inserting partsupp...';
END
$$;
\i sql/tpch/partsupp.sql
DO $$
BEGIN
    RAISE NOTICE 'Inserting supplier...';
END
$$;
\i sql/tpch/supplier.sql