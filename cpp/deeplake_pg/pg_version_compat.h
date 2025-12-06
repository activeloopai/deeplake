#pragma once

#define PG_VERSION_NUM_16 160000
#define PG_VERSION_NUM_17 170000
#define PG_VERSION_NUM_18 180000

extern "C" {
#if PG_VERSION_NUM >= PG_VERSION_NUM_18
#include <access/cmptype.h>
#include <commands/explain_format.h>
#endif
}

// Ordering operator compatibility
#if PG_VERSION_NUM >= PG_VERSION_NUM_18
#define PG_GET_OP_BTREE_INTERPRETATION get_op_index_interpretation
#define OpBtreeInterpretation OpIndexInterpretation
#define OpBtreeInterpretationGetStrategy(interp) ((interp)->cmptype)
#else
#define PG_GET_OP_BTREE_INTERPRETATION get_op_btree_interpretation
#define OpBtreeInterpretationGetStrategy(interp) ((interp)->strategy)
#endif

// Explain compatibility
#if PG_VERSION_NUM >= PG_VERSION_NUM_18
#define PG_EXPLAIN_PROPERTY_TEXT(name, value, es) ExplainPropertyValue(name, value, es)
#else
#define PG_EXPLAIN_PROPERTY_TEXT(name, value, es) ExplainPropertyText(name, value, es)
#endif

#if PG_VERSION_NUM >= PG_VERSION_NUM_18
using StrategyCompareType = CompareType;
#else
using StrategyCompareType = int16_t;
#endif