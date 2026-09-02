(() => {
  "use strict";

  const data = window.STYLEMATCH_METHOD_FLOW;
  if (!data) return;

  const nodes = new Map(data.nodes.map(node => [node.id, node]));
  const primary = ["query", "input_guard", "style_encoder", "retrieval", "verification", "result"];
  const support = [
    ["topic_encoder", "Affinity context only — never Style Match rank"],
    ["profile_memory", "Read-only evidence base for exact retrieval"],
    ["evidence", "Explanation and decade view — never retroactive ranking"],
  ];
  const advantages = [
    [
      "Can style travel across languages?",
      "One multilingual authorship space compares form; target-language grouping keeps score scales valid.",
    ],
    [
      "Why not translate every passage first?",
      "Original-text retrieval preserves syntax, punctuation, honorifics, and register. Translation is an ablation.",
    ],
    [
      "How is topic kept separate from style?",
      "Cross-work positives and context-matched negatives train style; a separate encoder reports topic.",
    ],
    [
      "How can one author contain several voices?",
      "Centroids capture the core voice; source prototypes preserve work-level variation.",
    ],
  ];

  function element(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text != null) value.textContent = text;
    return value;
  }

  function answerWithBreaks(answer) {
    const root = element("p", "");
    const lines = answer.split("; ");
    lines.forEach((line, index) => {
      root.append(document.createTextNode(`${line}${index < lines.length - 1 ? ";" : ""}`));
      if (index < lines.length - 1) root.append(document.createElement("br"));
    });
    return root;
  }

  function card(id, role) {
    const node = nodes.get(id);
    const root = element("article", `method-card flow-${id}${node.style === "dashed" ? " is-pending" : ""}`);
    root.dataset.category = node.category;
    root.append(
      element("div", "method-card-eyebrow", node.eyebrow),
      element("h3", "", node.label),
      element("p", "method-card-signature", node.signature),
      element("p", "method-card-detail", node.detail),
      element("span", "method-card-status", node.status.replaceAll("_", " ")),
    );
    if (role) root.append(element("strong", "support-role", role));
    return root;
  }

  function render(host) {
    const root = element("div", "method-static");
    const heading = element("header", "method-flow-heading");
    heading.append(
      element("div", "method-flow-kicker", `System map · evidence contracts · v${data.meta.version}`),
      element("h2", "", "From passage to accountable result"),
    );

    const advantageGrid = element("section", "method-advantage-grid");
    advantages.forEach(([question, answer]) => {
      const item = element("article", "method-advantage");
      item.append(
        element("h3", "", question),
        answerWithBreaks(answer),
      );
      advantageGrid.append(item);
    });

    const main = element("div", "primary-flow");
    primary.forEach(id => main.append(card(id)));

    const supportHeading = element("div", "support-heading", "Separate support planes");
    const supportFlow = element("div", "support-flow");
    support.forEach(([id, role]) => supportFlow.append(card(id, role)));

    root.append(heading, advantageGrid, main, supportHeading, supportFlow);
    host.replaceChildren(root);
  }

  document.querySelectorAll("[data-method-static]").forEach(render);
})();
