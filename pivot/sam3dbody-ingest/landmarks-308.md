# Landmarks appendix — mhr70 (70) and the 238 face rows of the 308 mapping

*Companion table for [metadata-objects.md](metadata-objects.md). Both enumerations were extracted read-only; evidence and commands are in that report.*

## 1. `mhr70` — all 70 keypoints, index order (`sam_3d_body/metadata/mhr70.py`)

Aliases: `mhr_names` uses hyphens (`right-thumb-tip`), `original_keypoint_info` underscores (`right_thumb_tip`), `keypoint_info` a third hand convention (`right_thumb4`, `right_forefinger4`, …) with `swap` partners and a stale comment at id 45. Indices 15–20 are `*_toe` in `keypoint_info` but `*_toe_tip` in `original_keypoint_info`. **Use indices, never names.**

```
 0 nose            1 left-eye        2 right-eye       3 left-ear        4 right-ear      5 left-shoulder
 6 right-shoulder  7 left-elbow      8 right-elbow     9 left-hip       10 right-hip     11 left-knee
12 right-knee     13 left-ankle     14 right-ankle    15 left-big-toe-tip 16 left-small-toe-tip 17 left-heel
18 right-big-toe-tip 19 right-small-toe-tip 20 right-heel 21 right-thumb-tip 22 right-thumb-first-joint
23 right-thumb-second-joint 24 right-thumb-third-joint 25 right-index-tip … 40 right-pinky-third-joint
41 right-wrist    42 left-thumb-tip … 61 left-pinky-third-joint  62 left-wrist  63 left-olecranon
64 right-olecranon 65 left-cubital-fossa 66 right-cubital-fossa 67 left-acromion 68 right-acromion 69 neck
```

Groups declared in `pose_info`: 17 `body_keypoint_names` + 6 `foot_*` + 20 `left_hand_*` + 20 `right_hand_*` + 7 `extra_*` (`neck`, 2 olecranon, 2 cubital fossa, 2 acromion) = 70.

## 2. Rows 70–307 — the 238 non-teeth face keypoints of the 308 mapping

Row = `70 + i`, where *i* indexes the Sociopticon/Sapiens 308 format's `face_keypoint_names` with the 36 `teeth_1..teeth_36` entries removed. Format ids 70–343 → 238 rows. Justification (structural + geometric, incl. the 100 % `l_`/`r_` x-sign test) is in metadata-objects.md §3.3.

```
70:center_of_glabella · 71:center_of_nose_root · 72:tip_of_nose_bridge · 73:midpoint_1_of_nose_bridge · 74:midpoint_2_of_nose_bridge
75:midpoint_3_of_nose_bridge · 76:center_of_labiomental_groove · 77:tip_of_chin · 78:upper_startpoint_of_r_eyebrow · 79:lower_startpoint_of_r_eyebrow
80:end_of_r_eyebrow · 81:upper_midpoint_1_of_r_eyebrow · 82:lower_midpoint_1_of_r_eyebrow · 83:upper_midpoint_2_of_r_eyebrow · 84:upper_midpoint_3_of_r_eyebrow
85:lower_midpoint_2_of_r_eyebrow · 86:lower_midpoint_3_of_r_eyebrow · 87:upper_startpoint_of_l_eyebrow · 88:lower_startpoint_of_l_eyebrow · 89:end_of_l_eyebrow
90:upper_midpoint_1_of_l_eyebrow · 91:lower_midpoint_1_of_l_eyebrow · 92:upper_midpoint_2_of_l_eyebrow · 93:upper_midpoint_3_of_l_eyebrow · 94:lower_midpoint_2_of_l_eyebrow
95:lower_midpoint_3_of_l_eyebrow · 96:l_inner_end_of_upper_lash_line · 97:l_outer_end_of_upper_lash_line · 98:l_centerpoint_of_upper_lash_line · 99:l_midpoint_2_of_upper_lash_line
100:l_midpoint_1_of_upper_lash_line · 101:l_midpoint_6_of_upper_lash_line · 102:l_midpoint_5_of_upper_lash_line · 103:l_midpoint_4_of_upper_lash_line · 104:l_midpoint_3_of_upper_lash_line
105:l_outer_end_of_upper_eyelid_line · 106:l_midpoint_6_of_upper_eyelid_line · 107:l_midpoint_2_of_upper_eyelid_line · 108:l_midpoint_5_of_upper_eyelid_line · 109:l_centerpoint_of_upper_eyelid_line
110:l_midpoint_4_of_upper_eyelid_line · 111:l_midpoint_1_of_upper_eyelid_line · 112:l_midpoint_3_of_upper_eyelid_line · 113:l_midpoint_6_of_upper_crease_line · 114:l_midpoint_2_of_upper_crease_line
115:l_midpoint_5_of_upper_crease_line · 116:l_centerpoint_of_upper_crease_line · 117:l_midpoint_4_of_upper_crease_line · 118:l_midpoint_1_of_upper_crease_line · 119:l_midpoint_3_of_upper_crease_line
120:r_inner_end_of_upper_lash_line · 121:r_outer_end_of_upper_lash_line · 122:r_centerpoint_of_upper_lash_line · 123:r_midpoint_1_of_upper_lash_line · 124:r_midpoint_2_of_upper_lash_line
125:r_midpoint_3_of_upper_lash_line · 126:r_midpoint_4_of_upper_lash_line · 127:r_midpoint_5_of_upper_lash_line · 128:r_midpoint_6_of_upper_lash_line · 129:r_outer_end_of_upper_eyelid_line
130:r_midpoint_3_of_upper_eyelid_line · 131:r_midpoint_1_of_upper_eyelid_line · 132:r_midpoint_4_of_upper_eyelid_line · 133:r_centerpoint_of_upper_eyelid_line · 134:r_midpoint_5_of_upper_eyelid_line
135:r_midpoint_2_of_upper_eyelid_line · 136:r_midpoint_6_of_upper_eyelid_line · 137:r_midpoint_3_of_upper_crease_line · 138:r_midpoint_1_of_upper_crease_line · 139:r_midpoint_4_of_upper_crease_line
140:r_centerpoint_of_upper_crease_line · 141:r_midpoint_5_of_upper_crease_line · 142:r_midpoint_2_of_upper_crease_line · 143:r_midpoint_6_of_upper_crease_line · 144:l_inner_end_of_lower_lash_line
145:l_outer_end_of_lower_lash_line · 146:l_centerpoint_of_lower_lash_line · 147:l_midpoint_2_of_lower_lash_line · 148:l_midpoint_1_of_lower_lash_line · 149:l_midpoint_6_of_lower_lash_line
150:l_midpoint_5_of_lower_lash_line · 151:l_midpoint_4_of_lower_lash_line · 152:l_midpoint_3_of_lower_lash_line · 153:l_outer_end_of_lower_eyelid_line · 154:l_midpoint_6_of_lower_eyelid_line
155:l_midpoint_2_of_lower_eyelid_line · 156:l_midpoint_5_of_lower_eyelid_line · 157:l_centerpoint_of_lower_eyelid_line · 158:l_midpoint_4_of_lower_eyelid_line · 159:l_midpoint_1_of_lower_eyelid_line
160:l_midpoint_3_of_lower_eyelid_line · 161:r_inner_end_of_lower_lash_line · 162:r_outer_end_of_lower_lash_line · 163:r_centerpoint_of_lower_lash_line · 164:r_midpoint_1_of_lower_lash_line
165:r_midpoint_2_of_lower_lash_line · 166:r_midpoint_3_of_lower_lash_line · 167:r_midpoint_4_of_lower_lash_line · 168:r_midpoint_5_of_lower_lash_line · 169:r_midpoint_6_of_lower_lash_line
170:r_outer_end_of_lower_eyelid_line · 171:r_midpoint_3_of_lower_eyelid_line · 172:r_midpoint_1_of_lower_eyelid_line · 173:r_midpoint_4_of_lower_eyelid_line · 174:r_centerpoint_of_lower_eyelid_line
175:r_midpoint_5_of_lower_eyelid_line · 176:r_midpoint_2_of_lower_eyelid_line · 177:r_midpoint_6_of_lower_eyelid_line · 178:tip_of_nose · 179:bottom_center_of_nose
180:r_outer_corner_of_nose · 181:l_outer_corner_of_nose · 182:inner_corner_of_r_nostril · 183:outer_corner_of_r_nostril · 184:upper_corner_of_r_nostril
185:inner_corner_of_l_nostril · 186:outer_corner_of_l_nostril · 187:upper_corner_of_l_nostril · 188:r_outer_corner_of_mouth · 189:l_outer_corner_of_mouth
190:center_of_cupid_bow · 191:center_of_lower_outer_lip · 192:midpoint_1_of_upper_outer_lip · 193:midpoint_2_of_upper_outer_lip · 194:midpoint_1_of_lower_outer_lip
195:midpoint_2_of_lower_outer_lip · 196:midpoint_3_of_upper_outer_lip · 197:midpoint_4_of_upper_outer_lip · 198:midpoint_5_of_upper_outer_lip · 199:midpoint_6_of_upper_outer_lip
200:midpoint_3_of_lower_outer_lip · 201:midpoint_4_of_lower_outer_lip · 202:midpoint_5_of_lower_outer_lip · 203:midpoint_6_of_lower_outer_lip · 204:r_inner_corner_of_mouth
205:l_inner_corner_of_mouth · 206:center_of_upper_inner_lip · 207:center_of_lower_inner_lip · 208:midpoint_1_of_upper_inner_lip · 209:midpoint_2_of_upper_inner_lip
210:midpoint_1_of_lower_inner_lip · 211:midpoint_2_of_lower_inner_lip · 212:midpoint_3_of_upper_inner_lip · 213:midpoint_4_of_upper_inner_lip · 214:midpoint_5_of_upper_inner_lip
215:midpoint_6_of_upper_inner_lip · 216:midpoint_3_of_lower_inner_lip · 217:midpoint_4_of_lower_inner_lip · 218:midpoint_5_of_lower_inner_lip · 219:midpoint_6_of_lower_inner_lip
220:l_top_end_of_inferior_crus · 221:l_top_end_of_superior_crus · 222:l_start_of_antihelix · 223:l_end_of_antihelix · 224:l_midpoint_1_of_antihelix
225:l_midpoint_1_of_inferior_crus · 226:l_midpoint_2_of_antihelix · 227:l_midpoint_3_of_antihelix · 228:l_point_1_of_inner_helix · 229:l_point_2_of_inner_helix
230:l_point_3_of_inner_helix · 231:l_point_4_of_inner_helix · 232:l_point_5_of_inner_helix · 233:l_point_6_of_inner_helix · 234:l_point_7_of_inner_helix
235:l_highest_point_of_antitragus · 236:l_bottom_point_of_tragus · 237:l_protruding_point_of_tragus · 238:l_top_point_of_tragus · 239:l_start_point_of_crus_of_helix
240:l_deepest_point_of_concha · 241:l_tip_of_ear_lobe · 242:l_midpoint_between_22_15 · 243:l_bottom_connecting_point_of_ear_lobe · 244:l_top_connecting_point_of_helix
245:l_point_8_of_inner_helix · 246:r_top_end_of_inferior_crus · 247:r_top_end_of_superior_crus · 248:r_start_of_antihelix · 249:r_end_of_antihelix
250:r_midpoint_1_of_antihelix · 251:r_midpoint_1_of_inferior_crus · 252:r_midpoint_2_of_antihelix · 253:r_midpoint_3_of_antihelix · 254:r_point_1_of_inner_helix
255:r_point_8_of_inner_helix · 256:r_point_3_of_inner_helix · 257:r_point_4_of_inner_helix · 258:r_point_5_of_inner_helix · 259:r_point_6_of_inner_helix
260:r_point_7_of_inner_helix · 261:r_highest_point_of_antitragus · 262:r_bottom_point_of_tragus · 263:r_protruding_point_of_tragus · 264:r_top_point_of_tragus
265:r_start_point_of_crus_of_helix · 266:r_deepest_point_of_concha · 267:r_tip_of_ear_lobe · 268:r_midpoint_between_22_15 · 269:r_bottom_connecting_point_of_ear_lobe
270:r_top_connecting_point_of_helix · 271:r_point_2_of_inner_helix · 272:l_center_of_iris · 273:l_border_of_iris_3 · 274:l_border_of_iris_midpoint_1
275:l_border_of_iris_12 · 276:l_border_of_iris_midpoint_4 · 277:l_border_of_iris_9 · 278:l_border_of_iris_midpoint_3 · 279:l_border_of_iris_6
280:l_border_of_iris_midpoint_2 · 281:r_center_of_iris · 282:r_border_of_iris_3 · 283:r_border_of_iris_midpoint_1 · 284:r_border_of_iris_12
285:r_border_of_iris_midpoint_4 · 286:r_border_of_iris_9 · 287:r_border_of_iris_midpoint_3 · 288:r_border_of_iris_6 · 289:r_border_of_iris_midpoint_2
290:l_center_of_pupil · 291:l_border_of_pupil_3 · 292:l_border_of_pupil_midpoint_1 · 293:l_border_of_pupil_12 · 294:l_border_of_pupil_midpoint_4
295:l_border_of_pupil_9 · 296:l_border_of_pupil_midpoint_3 · 297:l_border_of_pupil_6 · 298:l_border_of_pupil_midpoint_2 · 299:r_center_of_pupil
300:r_border_of_pupil_3 · 301:r_border_of_pupil_midpoint_1 · 302:r_border_of_pupil_12 · 303:r_border_of_pupil_midpoint_4 · 304:r_border_of_pupil_9
305:r_border_of_pupil_midpoint_3 · 306:r_border_of_pupil_6 · 307:r_border_of_pupil_midpoint_2
```

Category counts: nose 9 · eyebrows 18 · eyes/lash/lid 32 · mouth+lips 31 · chin 2 · ears 52 · iris 18 · pupil 18 · other (glabella, forehead/temple contour) 58.

**No jaw keypoint exists** in the 344/308 format (names audited). Jaw proxies: `77 tip_of_chin`, `76 center_of_labiomental_groove`, lower-lip rows 194/200–203/210/216–219.
