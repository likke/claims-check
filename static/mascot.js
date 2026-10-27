/* Pure mascot state mapping, shared with the review page and its tests. */
(function (root) {
  const frameForState = (state, severe = false) => severe ? 'hide-1-jump' : ({
    watching: 'popcorn', cur: 'prop-magnifier', worried: 'prop-clipboard',
    shocked: 'prop-redcard', unsure: 'confused', happy: 'cool', done: 'party'
  }[state] || 'popcorn');

  function isSevere(note, notes, pack) {
    if (!note?.claim) return false;
    const claim = note.claim;
    const first = claim.rounds?.[0] || {};
    const text = `${claim.quote || ''} ${first.line || ''} ${first.reason || ''}`;
    const fdaRule = pack === 'FDA beauty' && first.judgement === 'rule';
    const cureTreat = /\b(cure|treat|heal|prevent)\w*\b/i.test(text);
    const failures = (notes || []).filter(n => n.claim?.status === 'fail' && Number.isFinite(n.claim.start));
    const clustered = failures.some(a => failures.some(b => a !== b && Math.abs(a.claim.start - b.claim.start) <= 10));
    return fdaRule || cureTreat || clustered;
  }

  root.MascotStates = {frameForState, isSevere};
})(window);
