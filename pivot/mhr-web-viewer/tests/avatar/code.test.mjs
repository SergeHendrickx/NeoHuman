import test from "node:test";
import assert from "node:assert/strict";
import {
  AVATAR_CODE_VERSION,
  packAvatarData,
  unpackAvatarCode,
  validateAvatarData,
} from "../../static/js/avatarCodec.js";

const sample = () => ({
  v: 1,
  app: "neohuman",
  params: { identity: [0, 0.5, -1], expression: [0.125], pose: [0, -0.75] },
  scale: 1.05,
  look: { skin: "#bf8a5f", gaze: [8, -2] },
});

test("pack/unpack round-trips avatar data into URL-safe text", async () => {
  const code = await packAvatarData(sample());
  assert.match(code, /^[A-Za-z0-9_-]+$/);
  assert.deepEqual(await unpackAvatarCode(code), sample());
});

test("unpack rejects garbage and empty input", async () => {
  await assert.rejects(() => unpackAvatarCode("!!!not-a-code!!!"), /avatar code/i);
  await assert.rejects(() => unpackAvatarCode(""), /avatar code/i);
});

test("validate checks version and group lengths", () => {
  const counts = { identity: 3, expression: 1, pose: 2 };
  validateAvatarData(sample(), counts);
  assert.throws(() => validateAvatarData({ ...sample(), v: 99 }, counts), /version/i);
  const short = sample();
  short.params = { identity: [0], expression: [0.125], pose: [0, -0.75] };
  assert.throws(() => validateAvatarData(short, counts), /identity/i);
});
