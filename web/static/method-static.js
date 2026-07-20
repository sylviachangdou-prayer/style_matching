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

  function element(tag, className, text) {
    const value = document.createElement(tag);
    if (className) value.className = className;
    if (text != null) value.textContent = text;
    return value;
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
      element("div", "method-flow-kicker", "System map · evidence contracts · v1.0"),
      element("h2", "", "From passage to accountable result"),
      element("p", "", "The upper route is the complete author-ranking path. Supporting measurements stay below the line so their meanings cannot be visually or statistically collapsed."),
    );

    const main = element("div", "primary-flow");
    primary.forEach(id => main.append(card(id)));

    const supportHeading = element("div", "support-heading", "Separate support planes");
    const supportFlow = element("div", "support-flow");
    support.forEach(([id, role]) => supportFlow.append(card(id, role)));

    root.append(heading, main, supportHeading, supportFlow);
    host.replaceChildren(root);
  }

  document.querySelectorAll("[data-method-static]").forEach(render);
})();
