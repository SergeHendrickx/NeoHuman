from collections import Counter
from html.parser import HTMLParser
import re


class C(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.errs = []
        self.void = {"input", "br", "hr", "img", "meta", "link", "source"}

    def handle_starttag(self, tag, attrs):
        if tag not in self.void:
            self.stack.append((tag, self.getpos()))

    def handle_endtag(self, tag):
        if tag in self.void:
            return
        if self.stack and self.stack[-1][0] == tag:
            self.stack.pop()
        else:
            self.errs.append(f"mismatch </{tag}> at {self.getpos()}")


src = open("static/index.html", encoding="utf-8").read()
c = C()
c.feed(src)
print("balance errors:", c.errs if c.errs else "none")
print("unclosed:", [(t, p) for t, p in c.stack] if c.stack else "none")
ids = re.findall(r'id="([^"]+)"', src)
dupes = {k: v for k, v in Counter(ids).items() if v > 1}
print("duplicate ids:", dupes if dupes else "none")
print("total ids:", len(ids))
