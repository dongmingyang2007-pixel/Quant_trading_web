const assert = require('node:assert/strict');
const { test } = require('node:test');

require('../static/trading/js/market_insights_state.js');

const { mergeRankItems, toggleSortState } = global.MarketInsightsState || {};

test('mergeRankItems dedupes by symbol + exchange', () => {
  const existing = [
    { symbol: 'SLE', exchange: 'NYSE' },
    { symbol: 'UONE', exchange: 'NASDAQ' },
  ];
  const incoming = [
    { symbol: 'SLE', exchange: 'NYSE' },
    { symbol: 'SLE', exchange: 'NYSE' },
    { symbol: 'UONE', exchange: 'NASDAQ' },
    { symbol: 'UONEK', exchange: 'NASDAQ' },
  ];
  const result = mergeRankItems(existing, incoming);
  assert.equal(result.merged.length, 3);
  assert.equal(result.appended.length, 1);
  assert.equal(result.appended[0].symbol, 'UONEK');
});

test('toggleSortState cycles default → desc → asc → default', () => {
  assert.equal(toggleSortState('default'), 'desc');
  assert.equal(toggleSortState('desc'), 'asc');
  assert.equal(toggleSortState('asc'), 'default');
});
