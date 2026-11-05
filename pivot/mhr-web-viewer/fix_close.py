P = "static/index.html"
lines = open(P, encoding="utf-8").read().splitlines(keepends=True)
ai = next(i for i, l in enumerate(lines) if l == "    </aside>\n" and i > 800)
assert lines[ai - 1].strip() == "</div>", lines[ai - 1]
lines[ai:ai] = ["      </section>\n"]
open(P, "w", encoding="utf-8").write("".join(lines))
print("actor-card close restored")
