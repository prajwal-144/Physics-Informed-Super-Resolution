# Presenter's guide — what to say, in plain words

For the 29-slide deck `DeepLense_progress_review.pptx`. Nothing here is jargon for its own sake.
Read Part 0 and Part 1 once, then Part 3 the morning of the meeting. Part 2 is what to say on each
slide; Part 5 is what *not* to say.

**You do not need to memorise this.** The slides carry the numbers. What you need to carry is the
*story*, and the story is nine sentences long — that is Part 0.

---

# PART 0 — the whole thing in nine sentences

1. A galaxy sits behind a heavy blob of mass, the mass bends the light, and we see a smeared ring
   or arc instead of the galaxy.
2. We want to undo that bending and get the galaxy back, sharper than the camera could ever have
   photographed it — and we have no sharp picture to learn from, because nobody has one.
3. So instead of learning from sharp pictures, we learn from **physics**: whatever galaxy we guess,
   we push it back through the bending, blur it the way the telescope blurs, chop it into detector
   pixels, and check whether it matches the one photograph we actually have.
4. The old version of this project kept the bending fixed and simple — a perfect circle — and read
   its size from the answer key. The reconstructed galaxy came out **eleven times too big**, and no
   amount of tuning fixed it.
5. We replaced that with a proper lens model with six knobs, and we **fit those knobs to each
   photograph individually**. The galaxy now comes out the right size: **1.08 times** instead of
   eleven.
6. Because the recovered galaxy is a smooth mathematical formula rather than a picture, we can
   redraw it at any level of fineness we like — and we checked that redrawing it eight times finer
   introduces no error, so the fine detail is real, not invented.
7. We then trained a neural network to guess those same knobs in one shot. It is fast but it
   systematically under-estimates how squashed the lens is — for a mathematical reason, not because
   it needs more training. **Used as a starting point for the fit instead of as the answer, it
   removes two thirds of the work and the result is slightly better.**
8. Then we brought back the original project's own network design — un-warp the photo first, then
   sharpen it with a fully convolutional network — running on our corrected lens. It gives the best
   free-form galaxy reconstruction in the project, roughly half the error of everything before it.
9. Finally we tested the central assumption — that the lens's magnification is what makes the extra
   sharpness possible. It is true, and measurable: we can inject a blob smaller than one detector
   pixel and get 17–23% of it back. But **how many photons you collected matters about three times
   more than how much the lens magnified**, and that single fact explains why four separate attempts
   to exploit magnification all came out flat.

If you can say those nine, you can present the deck.

---

# PART 1 — the vocabulary, in plain words

Say these in your own words. Nobody will ask you to define them formally, but if you are fluent
here you will never be stuck.

### The physics

**Gravitational lens.** A galaxy or cluster in the foreground whose gravity bends the light of
something behind it. Like looking through the base of a wine glass: you see the thing behind it,
warped, stretched, sometimes duplicated.

**Source.** The background galaxy — the thing we want back.

**Image / observation.** What the telescope actually recorded: the warped ring or arcs.

**Einstein radius (θ_E).** The size of the ring. Bigger ring = more mass in the lens. It is the
single most important number describing a lens. Ours are about 1.3 arcseconds across.

**Ellipticity.** How squashed the lens is, rather than perfectly round. Two numbers, because it has
both an amount and a direction. **This turned out to be the thing that matters most** — see Part 3,
Q2.

**External shear.** A gentle, uniform stretch caused by mass *outside* the picture — a neighbouring
galaxy, a group down the line of sight. Two more numbers. It adds a stretch but no mass of its own.

**Radial slope.** How quickly the lens's mass thins out as you move away from its centre. One
number. In practice it sits very close to a standard value, so it is the hardest one to pin down.

**Magnification.** How much the lens spreads a small patch of the background galaxy across the sky.
Where magnification is high, a tiny patch gets smeared across many camera pixels — **which is
exactly why we can recover detail finer than a camera pixel.** The sky did the zooming for us.

> **Important, and worth saying out loud if magnification comes up:** magnification changes how much
> *area* something covers, not how *bright* it is. We never multiply brightness by magnification
> anywhere. Getting this wrong is a classic error in this field.

**Critical curve / the ring.** The circle (not quite a circle) where magnification blows up. For a
perfectly round lens it is a true circle; for a squashed one its radius wobbles around the ring.
**That wobble is a great test**, because a round lens has zero wobble — so recovering the wobble
proves you recovered the squashing.

**Sérsic profile.** The standard mathematical shape astronomers use for a galaxy's brightness: a
bright centre fading outwards, with knobs for how big, how concentrated, how squashed, and where it
sits. Seven numbers in total. Real galaxies are roughly this shape; the simulated galaxies in our
dataset are *exactly* this shape, which matters (Part 3, Q7).

**PSF — the telescope blur.** Point a perfect telescope at a single star and you still get a small
smudge, not a dot. That smudge is the PSF. Every image is the true sky blurred by it. We
**measured** ours instead of assuming it: 0.18–0.20 arcseconds wide, not the 0.10 that had been
assumed, and with wider outer edges than a simple bell curve.

### The method

**Forward model.** Our software model of what the telescope does to a galaxy:
**galaxy → bent by the lens → blurred by the telescope → chopped into detector pixels.** In that
order — blurring happens in the sky, chopping happens inside the camera.

**Chi-squared (χ²).** A score for how badly a model misses the data. For every pixel: take
(model − data), divide by how noisy that pixel is, square it, add them all up. Dividing by the noise
is the important part — it means a pixel we measured badly counts less. Lower is better, and
minimising it is the same as choosing the most likely explanation of the data.

**Fitting.** Turning knobs until the score is as low as it can get.

**Levenberg–Marquardt (and what we actually use).** A classic, very well-known recipe for turning
knobs on a smooth problem. It blends two strategies: near the answer it takes big confident jumps
(fast); far from the answer it takes small careful steps (safe); and it switches between them
automatically depending on whether the last step helped. We use SciPy's bounded version of this
idea, called *trust-region reflective*, which additionally refuses to let a knob go somewhere
physically impossible — like a negative ring size. **Say "bounded least-squares fitting" and you are
safe.**

**Rank correlation (Spearman).** "Does it get the *order* right?" Sort the 2,000 images by our
answer, sort them by the truth, and see whether the two orderings agree. +1 is perfect, 0 is
useless. We use it because a constant offset — everything 3% too big — is a calibration problem you
can fix, while a scrambled order means you have nothing. Rank correlation ignores the first and
catches the second.

**Size ratio.** Our reconstructed galaxy's size divided by the true galaxy's size. Target 1. This is
the headline number of the whole project, because it is the one that exposed the old pipeline
(10.6–12.8) and the one that shows the new one works (1.08).

**Back-projection.** Take the observed photograph, trace every pixel backwards through the lens, and
drop its brightness where it came from. You get a rough, noisy, patchy picture that is already
roughly the galaxy. It is not a reconstruction — it is a *good starting picture* for a network to
sharpen.

**Source box.** The small window in the "un-warped" plane where we put a pixel grid. Yes — it is
just a grid (see Part 3, Q10).

**Correction map.** A small extra image added on top of the smooth galaxy shape, holding whatever
the smooth shape could not explain. See Part 3, Q4.

**Magnification gate.** A rule that deletes fine detail from the answer wherever the lens did not
magnify enough for that detail to be measurable. Not a suggestion the optimiser can ignore — the
detail is physically removed before the comparison happens.

---

# PART 2 — slide by slide

For each slide: **[SAY]** the two or three sentences that carry it, **[NUMBER]** the one figure
worth quoting, and **[IF ASKED]** the follow-up you are most likely to get.

---

### Slide 1 — Title

**[SAY]** "This covers the rewritten pipeline: fitting a lens and a galaxy to each image, a network
that does the same job in one pass, that network used as a starting point instead, and two newer
models built in the last two days."

---

### Slide 2 — The task and the one constraint

**[SAY]** "We want the background galaxy back, sharper than the camera recorded it. The problem is
that there is no sharp version to train against — you cannot re-observe the same lens with a better
telescope. So the training signal has to be physics: whatever galaxy we propose has to reproduce the
photograph after being lensed, blurred and pixelated."

**[IF ASKED] "Why not just simulate high-resolution pairs and train on those?"** Then the network
learns the simulator's conventions, not physics, and it has no reason to work on a real image. Also
we would have no way of knowing whether it worked, because on real data there is no sharp version to
check against.

---

### Slide 3 — What was measurably wrong at the start

**[SAY]** "The previous version kept the lens fixed, perfectly circular, and read its size from the
simulation's answer key. The reconstructed galaxy came out ten to thirteen times too large, and that
did not move at all across a hundred-fold sweep of the smoothing setting."

**[SAY, and this is the good bit]** "That is not a tuning failure — it is the *correct* thing for the
model to do given a wrong lens. If the lens is round but the real one is squashed, you cannot put
light at the right place around the ring no matter what galaxy you propose. So the best remaining
move is to inflate the galaxy until the predicted ring is fat enough to overlap the real arc
everywhere. Fattening the source is the optimal response to a broken forward model."

**[NUMBER]** A 3-pixel blur of the input scored **0.951** on the old headline metric, while the
exact, true, noiseless physical model scored **0.630**. A physical model cannot beat the truth — so
anything above 0.63 was fitting the noise.

**[IF ASKED] "How did nobody notice?"** Because every metric available at the time asked "does this
look like a galaxy?" — compact, single blob, smooth — and an eleven-times-too-large blob passes all
of those. The new dataset ships the true galaxy, so for the first time we could ask "is it *the*
galaxy?" instead.

---

### Slide 4 — The data

**[SAY]** "Roman-telescope-like simulations, 127 by 127 pixels. The important property is that every
file carries the complete answer key — the true galaxy, the true mass map, and every parameter. That
is what makes all the checking possible."

**[SAY THE CAVEAT — do not let them find it]** "One correction to earlier write-ups: every result I
show is from the axion class only. The data loader sorts all three classes into one list and then
truncates, so asking for three classes returned axion images. It does not change any number, but it
means generalisation across the classes is untested. It is a thirty-minute fix and a re-run."

**[IF ASKED] "Does that invalidate anything?"** No — training and validation are different splits, so
there is no leakage, and axion is the hardest class because it has the most substructure. If
anything the numbers are conservative.

---

### Slide 5 — The forward model

**[SAY]** "This is our model of what the telescope does. Four steps: the galaxy sits in the sky, the
lens bends it, the telescope blurs it, the detector chops it into pixels. The order matters — the
blur happens in the sky, the chopping happens inside the camera."

**[SAY — the key structural point of the whole project]** "The bending is computed from a formula
with six knobs, evaluated live, rather than looked up from a stored table. That is what makes the
lens *fittable*. In the old version the lens was a precomputed matrix on disk, and you cannot
optimise something you loaded from a file."

**[NUMBER]** The formula agrees with `lenstronomy`, the standard lensing library, to about **3 parts
in 10¹⁵**.

**[IF ASKED] "Why not just use lenstronomy?"** We do — as the reference we check against. But it is
NumPy-only, so it cannot run inside PyTorch and cannot give us gradients with respect to the lens
parameters, which is exactly what training needs. Re-implementing physics in a differentiable
framework and validating it against the reference is the standard approach in this area.

---

### Slide 6 — The training signal

**[SAY]** "We never compare against a sharp image, because there isn't one. We compare the *predicted
photograph* against the *actual photograph*, pixel by pixel, weighted by how noisy each pixel is."

**[SAY]** "The answer key is opened only by the scoring script, after a fit is finished and written to
disk. Nothing in any loss function has ever seen it."

**[IF ASKED] "Then where do the starting values come from — isn't that cheating?"** From the picture
itself. We walk outward at 72 angles, find the ring of peak brightness, and fit a simple wave to it.
The flat part gives the ring size to about half a pixel; the first wobble gives how far off-centre
the galaxy is. All measured from pixels.

---

### Slide 7 — Step 1: fit a lens and a galaxy to each image

**[SAY]** "Fourteen numbers per image: six for the lens, seven for the galaxy, one for the sky
background. We turn all fourteen until the predicted photograph matches the real one."

**[SAY]** "Fourteen unknowns against six thousand measurements is enormously over-determined, which is
why this part needs no smoothing assumptions at all — there is nothing for a prior to do."

**[SAY]** "We release the knobs in four stages, because letting all fourteen loose from a cold start
lets the fit trade galaxy shape against lens stretch before the ring size is even right. The last
stage has everything free, so nothing ends up pinned at a value the data didn't choose."

**[NUMBER]** About **0.8 seconds** and **480 model evaluations** per image.

---

### Slide 8 — The recovery table

**[SAY]** "Seven physical quantities, recovered from the pixels alone. Nothing in this table was
available to the fit."

**[SAY, pointing at the last column]** "The last column is rank correlation — does it order the two
thousand images correctly. Ring size 0.96, galaxy offset 0.91, galaxy size 0.92, galaxy
concentration 0.94, lens squashing 0.76."

**[SAY THE BIAS BEFORE THEY SPOT IT]** "Ring size comes out about 3.5% high and galaxy size about 10%
high. Both point the same way, and both are consistent with our telescope-blur model having
narrower outer edges than the real one. It is a known, single, fixable cause."

**[IF ASKED] "Why rank correlation and not the actual error?"** We show both — median error is the
third column. Rank correlation is there because a constant 3% offset is a calibration you can
correct, while a scrambled order means no information. The two numbers answer different questions.

---

### Slide 9 — Fitted against true

**[SAY]** "Tight diagonals for ring size, galaxy offset and galaxy size. Looser for the squashing. A
cloud for the radial slope, which the data barely constrain — the simulated lenses only span 1.90 to
2.20, so there is very little signal there."

---

### Slide 10 — The galaxy is the right size now

**[SAY]** "This is the number that exposed the old pipeline and the number that shows the new one
works. Reconstructed size divided by true size. Target 1. Old: between 10.6 and 12.8. New: 1.08."

---

### Slide 11 — Where the extra sharpness comes from

**[SAY]** "The recovered galaxy is seven numbers describing a smooth mathematical curve, not a
picture. Once we have those numbers we can draw the galaxy on any grid we like, however fine. That
is the super-resolution — we never learn what galaxies look like, we invert a calibrated physical
model and then draw its answer finely."

**[SAY — pre-empt the obvious objection]** "The obvious objection is that this is just interpolation
with extra steps. So we tested it: draw our galaxy and the true galaxy on the same fine grid, at one,
two, four and eight times finer, and compare. If we were inventing detail, the agreement would fall
apart as we went finer. It doesn't — the correlation moves by 0.0017 across an eight-fold range. The
limit is the fitted numbers, not the grid."

---

### Slide 12 — The pictures

**[SAY]** "Left is what the telescope saw. Second is our reconstruction at eight times finer. Third is
the truth at the same fineness. Fourth is the truth as stored in the dataset. Columns two and three
should look like the same object — and the first column is visibly coarse and noisy by comparison."

---

### Slide 13 — The lens is the magnifying element

**[SAY]** "This is the physical reason any of this is possible. Where the lens magnifies, a small patch
of the background galaxy is smeared across many camera pixels — so the sky has already over-sampled
it for us. We are not inventing resolution, we are recovering resolution the lens delivered."

**[NUMBER]** Measured across 2,000 images: the galaxy is stretched by a factor of **3.08** along the
arc and essentially not at all across it. So a grid up to about three times finer than the detector
is supported by the data. Both networks use about two times — inside that limit — and the code prints
a warning if you ask for more.

**[SAY]** "And magnification is never multiplied into a brightness anywhere. It changes area, not
brightness."

---

### Slide 14 — A check the fit could not pass by accident

**[SAY]** "The ring is only a true circle if the lens is perfectly round. If it is squashed, the ring's
radius wobbles as you go around it, and the size of that wobble depends on the squashing and the
external stretch together."

**[NUMBER]** Recovered wobble **0.540 arcsec** against a true **0.548**, and the ring's position is
right to **0.63 pixels**. Total magnification recovered to about **1%**.

**[SAY WHY IT MATTERS]** "A round lens has a wobble of exactly zero. So you cannot recover this by
accident, and you cannot recover it by getting the ring size right — you have to have recovered the
squashing and the external stretch *jointly*."

---

### Slide 15 — Step 2: a network that predicts the same fourteen numbers

**[SAY]** "The fit takes about 0.8 seconds per image. Upcoming surveys will find of order a hundred
thousand lenses. So we trained a small network to read the image and emit the same fourteen numbers,
plus a small correction patch."

**[SAY]** "It is trained with exactly the same rule — no sharp target, just physics. And every output
is squeezed through a bounded function so it physically cannot emit a negative ring size, which would
produce undefined values before there is any gradient to correct it."

**[SAY — the honest bit]** "The first full run failed, and it failed for one measurable reason: the
score was never normalised properly, so one bright image in a batch of sixteen was taking 54% of the
learning signal. The effective batch size was about two. Fixing the noise model fixed the run."

---

### Slide 16 — What the network gets right and wrong

**[SAY]** "Ring size, galaxy size — as good as the fit or better. Galaxy concentration and lens
squashing — much worse. The squashing comes out about four times too small."

**[SAY — this is the interesting part, take your time]** "That is not undertraining. We checked: the
network gets the *direction* of the squashing roughly right and the *amount* badly wrong, by the same
factor of about 0.22 in every direction-accuracy bin. Here is why. Squashing is a quantity with a
direction. If you are unsure of the direction, guessing a large amount pointed the wrong way is
punished more heavily than guessing almost nothing. A network trained on squared error is effectively
predicting an average over everything consistent with the image, so the safe answer is to shrink it
toward zero. The per-image fit has no such pressure — it answers for one image, not for an average.
Both are behaving exactly as their objectives say they should, and more training will not change it."

---

### Slide 17 — The ellipticity figure

**[SAY]** "Left: the network alone, visibly compressed toward zero. Right: the same network output used
as the starting point for a short fit — the compression is completely gone."

---

### Slide 18 — Use the network as a starting point

**[SAY]** "So we stopped asking the network to be the answer and started using it as the starting
point. Same fit, same bounds, same tolerances — the only difference is where it begins."

**[NUMBER]** From 480 model evaluations to **151**. From four stages to **one**. And the score is
**3228 against 3302** — slightly better, not slightly worse.

**[SAY]** "The four-stage schedule only exists because a cold start falls into bad minima. From the
network's start it is unnecessary — and the fit lands in a *better* place, not just faster. Recovery
improves on six of seven parameters, most of all on the squashing."

**[SAY]** "This is verified on 800 validation images, and then used for real on 6,000 training images
to supply the lenses that the next two models need."

---

### Slide 19 — Summary figure

**[SAY]** "Network alone, the fit alone, and the two combined, on parameter recovery and on agreement
with the true galaxy."

---

### Slide 20 — Step 3 (B4): put the data in the source plane first

**[SAY]** "The step-2 network squeezes the whole image down to a single summary vector and then has to
rebuild a spatial map out of it. That throws away *where* everything was. Its correction patch never
got above three or four percent of the galaxy's brightness."

**[SAY]** "So B4 removes that bottleneck twice over. First, un-warp the photograph — trace every pixel
back through the fitted lens and drop it where it came from. You get a rough, noisy picture that is
already in the right place. Then a fully convolutional network — no summarising step anywhere —
sharpens it to a two-times finer grid."

**[SAY THE CREDIT]** "This is the architecture from the original DeepLense grid-based work, running on
our fitted lens instead of a fixed circular one. So it is a controlled swap of exactly one component,
which makes the comparison clean."

**[SAY]** "Two modes: the network output is the whole galaxy, or the fitted smooth galaxy is added and
the network only predicts the *difference*."

---

### Slide 21 — The B4 picture

**[SAY]** "Column two is the network's input — the un-warped photograph. It looks noisy because it is
built from the observed pixels, and it is noisiest exactly where few light rays landed. That texture
is literally the inverse magnification map."

---

### Slide 22 — B4 results

**[SAY]** "Adding the fitted smooth galaxy and predicting only the difference is clearly the better
mode: error 0.041 against 0.053, and the gap to the smooth fit on data agreement drops from 1.6 times
to 1.08."

**[SAY]** "This is the best free-form reconstruction in the project — about half the error of both
earlier attempts."

**[SAY THE HONEST FRAMING]** "It does not beat the seven-number fit, and it should not be expected to:
the simulated galaxies are *exactly* the smooth shape that fit assumes, so that fit has the perfect
prior. What B4 does beat it on is size and peak — the bias measures. So the free-form part is removing
the smooth model's systematic error at the cost of adding a little scatter."

**[SAY — the sentence they will remember]** "And notice the free-form run explains the photograph
*better* while sitting *further* from the truth. It had more data, more epochs and a bigger window,
and it still ends up worse. Fitting the data better is not the same as recovering the source."

---

### Slide 23 — Lensed observation in, sharp galaxy out

**[SAY]** "Left, what the telescope saw. Right, the recovered galaxy on a grid twice as fine. No sharp
image was used anywhere in producing this."

---

### Slide 24 — Every stage

**[SAY]** "Left to right: the observation; the un-warped input; the true galaxy at the dataset's own
resolution; the true galaxy redrawn on our fine grid so the eye can compare like with like; our
reconstruction; and then our reconstruction pushed back through the lens, the blur and the detector —
which is what the score actually compares against."

**[IF ASKED] "Why does the 'true' galaxy look pixelated?"** Because the dataset stores it at the
*detector's* resolution — 23 by 23 pixels inside our window — while our output is 48 by 48. There is
no high-resolution truth anywhere in this dataset. That is the whole point of the project: if one
existed we would just train on it. Column four redraws the truth analytically at our resolution, but
that is for looking at only — every number is scored against the stored array.

---

### Slide 25 — Step 4 (B5): the magnification gate

**[SAY]** "B4 already used magnification twice — as an input, and as a weight in the smoothing penalty.
B5 tries something stronger: a hard rule. Where the magnification is below about two, the fine detail
is physically deleted from the answer and it falls back to the coarse grid. The model cannot claim
resolution the lens did not deliver, and unlike a penalty it cannot trade it away against fitting the
data better."

**[SAY THE RESULT PLAINLY]** "It changed essentially nothing, and marginally for the worse. That is
the fourth independent attempt to exploit magnification adaptively in this project, and the fourth
null result. I am reporting it as a result rather than dropping it — the idea has now been tried as a
prior in a linear solve, twice as a weight in the loss, and once as a hard architectural constraint."

---

### Slide 26 — Does magnification predict where it works?

**[SAY]** "Four nulls demanded an explanation, so we measured the thing directly, on an already-trained
model, with no retraining. We inject a blob three quarters of a detector pixel wide into the galaxy,
push it through the full pipeline — lens, blur, pixels, noise — reconstruct the same system with and
without it using the *same* noise, and measure how much comes back. 2,400 injections."

**[NUMBER 1]** "It does come back, at 17 to 23 percent of what we put in. So sub-pixel structure is
genuinely being recovered — that is the first direct evidence in this project that the
super-resolution is real."

**[NUMBER 2]** "Recovery is better where magnification is above two: 0.208 against 0.175. That is a
19 percent difference and it is statistically solid."

**[NUMBER 3 — the punchline]** "But magnification is not the main driver. Its correlation with
recovery is +0.056. Signal-to-noise correlates at +0.152 — about three times stronger. On this data
the binding constraint is how many photons you collected, not how much the lens magnified. That
single fact explains all four null results at once."

---

### Slide 27 — What is checked

**[SAY]** "Every piece of physics is checked against either the standard library or an independent
derivation, and the numbers are agreements, not opinions."

**[SAY]** "One thing worth pointing out: the scoring code is shared. The fit, the step-2 network and B4
are all scored by the same function on the same images. If they were scored by separate scripts, any
difference could be a difference in the *measurement* rather than in the *method*, and there would be
no way to tell which."

---

### Slide 28 — Limitations

**[SAY]** Read them out. Do not soften them. The five, in order: axion class only; the smooth model can
only make smooth shapes; magnification is derived from the fitted lens rather than measured
independently; the score does not reach its ideal value because the telescope-blur model is imperfect
in a measured way; and everything is simulation.

**[SAY on the blur one]** "The signature that it is a systematic and not just noise is that the score
gets *worse* on brighter images. Noise would not do that."

---

### Slide 29 — Next steps

**[SAY]** "In order: fix the class sampling and re-run. Then inject structure the smooth model cannot
represent — a clump, or a real galaxy photograph as the source — and show the free-form part recovers
what the smooth fit cannot. That is the single most convincing experiment left and it is about a day."

**[SAY]** "Then the second colour band. Every image has two, we use one, they correlate at 0.75, and
the lens is identical in both — so a two-band fit constrains one lens with twice the data. That is the
biggest untouched lever."

**[SAY]** "And the workshop paper, four pages, deadline the 29th."

---

# PART 3 — the twelve questions you are most likely to get

### Q1. "Why an elliptical power law rather than the simpler elliptical model?"

The simpler model (SIE) is the *same* model with the radial slope frozen at the standard value. Ours
(EPL) just leaves it free. Three reasons: the simulator generated the data with the slope free, so
freezing it would introduce a mismatch that the other parameters would absorb; it costs exactly one
extra number against six thousand measurements, so there is no conditioning cost; and if the data
really don't constrain it, the fit simply returns the standard value — which is what happens, our
median comes out at 2.000 against a true 2.053.

The honest addition: on *this* dataset the extra slope is worth very little. Measured, the ladder
runs 0.973 for the simple elliptical model, 0.983 with the free slope, and 0.997 once external shear
is added. So the slope buys +0.010 and the shear buys +0.024. **Squashing buys +0.10.** Keep the free
slope because it is nearly free and matches the simulator, but do not claim it as important.

### Q2. "Why external shear?"

Two reasons, one physical and one measured.

Physical: external shear represents mass *outside* the frame — a neighbouring galaxy, a group along
the line of sight. It adds a stretch with no mass of its own. Nearly every real lens needs it, and
leaving it out is a classic source of bias in lens modelling, because the fit will try to explain
that stretch by making the lens itself more squashed than it is.

Measured: with the true galaxy and the true lens shape but no shear, the model reproduces the
observed image at correlation 0.973, and one image in five falls below 0.9. Add the true external
shear and it goes to **0.997**, with only one image in fifty below 0.9. Also, the dataset was
generated with it — the answer key has an external-shear entry with a median of 0.033.

**One clarification if someone presses:** a squashed lens *also* produces shear, but that is internal
to the lens and is already described by the two squashing numbers. "External shear" is specifically
the separate, uniform contribution from outside. There is no third thing called "extended shear" in
this project.

### Q3. "What exactly are the 6,361 measured pixel values?"

The image is 127 by 127 = 16,129 pixels. We only fit inside a circle of radius 45 pixels around the
centre, and that circle contains **6,361 pixels**. Those are the measurements.

Why a circle: the arcs live within about 30 pixels of the centre. The corners of the frame are pure
noise, and including them would add thousands of measurements that constrain nothing while diluting
the score. The circle is still big enough to contain plenty of blank sky, so the sky-background knob
is still well determined.

So: **14 unknowns against 6,361 measurements** — over-determined by a factor of about 450. That
ratio is why this part of the pipeline needs no smoothing assumptions at all.

### Q4. "Why is there a correction map in B3, and what is it?"

**The problem it solves.** The smooth galaxy shape has seven knobs. Seven knobs can only make
smooth, single-blob shapes. If the real galaxy has a clump, a spiral arm, or a companion, no setting
of those seven knobs can produce it — the fit will smooth it away and you would never know.

**The naive fix and why it fails.** Throw away the smooth shape and use a free grid of pixels
instead. That was tried twice in this project. With 64,516 free pixels against about 4,300
informative measurements the answer came out eleven times too big. With 4,096 free pixels it came out
as speckle. There simply is not enough independent information in one photograph to determine
thousands of free numbers.

**The correction map is the compromise.** Keep the smooth shape for the part the physics can predict,
and add a small grid — 32 by 32 — for the part it cannot:

> galaxy = smooth shape (7 numbers) + correction patch (32 × 32 numbers)

Three things make it safe:

1. **It is bounded.** The patch is squeezed through a function that caps it at 30% of the galaxy's
   brightness. It can adjust the smooth shape; it cannot replace it.
2. **It is penalised**, and penalised *unevenly* — cheap where the lens magnified a lot and the data
   really do constrain fine detail, expensive where the lens magnified little and anything you draw
   is invention.
3. **It degrades gracefully.** If the patch learns nothing it goes to zero and you are left with
   exactly the smooth fit, which already works. **The model cannot do worse than the fit; it can only
   add what the smooth shape missed.**

**Why 32 by 32 specifically.** Four constraints intersect there. It makes 14 + 1,024 = 1,038 unknowns
against ~4,300 measurements, so still four times over-determined. Over a window of ±0.8 arcseconds it
gives a pixel 2.05 times finer than the detector — inside the 3.08 times the lens actually delivers,
so we are not claiming resolution that isn't there. The decoder's two doubling steps land on exactly
32 with no resizing. And it leaves about five light rays landing in each grid cell, just above the
level where the magnification map becomes mostly counting noise.

**What it actually did.** The patch settled at 3–4% of the galaxy's brightness, never hit its cap,
concentrated where the magnification was high, and improved the galaxy reconstruction by about 5%.
Small, honest, and in the right place — which is what you want when the true galaxies are genuinely
smooth.

### Q5. "Did you write the fitting algorithm yourself?"

No — we use **SciPy's** `scipy.optimize.least_squares` with `method="trf"`. That is a mature, widely
used implementation. What we wrote is the part that is ours: the physics model, the residual function
(the thing being minimised), the parameter bounds, and the staged release schedule. It lives in
**`superres/fit_per_image.py`**, and `superres/refine_pathb.py` imports the same pieces so the warm
and cold fits are guaranteed identical apart from the starting point.

**Be precise about the name.** "Levenberg–Marquardt" is the classic recipe; what we call is
*trust-region reflective*, which is the bounded relative of it. We need the bounded version because
some knobs have hard physical limits. Saying **"bounded nonlinear least squares"** is accurate and
safe.

### Q6. "Did you write the loss function or use a library one?"

**Written from scratch, everywhere.** There is no `MSELoss` or any other library loss anywhere in the
project — I checked. The score is written out explicitly:

- in the fit: `(model(v)[mask] - data) / sigma` inside the residual function
- in the networks: `chi2 = (((pred - X) / sigma)[:, mask] ** 2).mean()`

The extra terms — the smoothness penalty and the magnification-weighted amplitude penalty — are also
hand-written, because both needed properties no standard loss has: the noise term includes a measured
systematic floor, and the penalty strength is expressed as a *fraction of the score* rather than an
absolute number, so it cannot silently become millions of times too small when the score changes
scale. That is exactly the bug that broke the first network run.

### Q7. "Your free-form model loses to the seven-parameter fit. Isn't that a failure?"

No, and it is important to say why. **The simulated galaxies are exactly the smooth shape that the
seven-parameter fit assumes.** That fit therefore has the perfect prior and only seven unknowns — it
is the hardest possible baseline, and no free-form model should be expected to beat it on this data.

What the free-form model *does* do: it comes within a factor of 1.6 on error while assuming no
particular galaxy shape at all, it is better calibrated on size and peak, and it is the form you
actually need for real galaxies, where no smooth family applies. And we have now shown directly that
it recovers injected structure smaller than a detector pixel — which the smooth fit, by construction,
cannot.

### Q8. "Why does the score not reach 1?"

Because our telescope-blur model is imperfect in a measured way. The arcs are bright — up to ten
thousand times the background noise — so a **one percent** error in the shape of the blur is a
fifty-sigma error in a single pixel. The signature that this is a systematic rather than noise is
that the score gets *worse* on brighter images; noise would not do that. For reference, on the same
images a flat grey image scores 37,511, a blurred copy of the input scores about 2,800–4,100, and our
physical fit scores 3,085 — while also recovering seven physical parameters.

And part of the residual is *supposed* to be there: it is the dark-matter substructure, which is
about 135 times the background noise level and which a smooth model cannot produce by construction.

### Q9. "Why do you only freeze six lens parameters in B4?"

Small correction to the premise: **B4 freezes more than six.** It reads the whole fourteen-number fit
and uses:

- the **six lens numbers** — these define the ray-tracing, both for un-warping the photo and for
  re-lensing the answer;
- the **galaxy brightness** — so the network only has to learn a shape, not a scale, which matters
  because the images vary 260-fold in brightness;
- the **sky background** — it enters the predicted photograph;
- and in the better mode, the **whole smooth galaxy shape**, which is added underneath and also fed
  in as an input channel.

So it is six in free-form mode plus brightness and background, and all fourteen in the mode we
actually recommend.

**Why freeze at all?** Because B4 exists to answer one question: *given a correct lens, which decoder
reconstructs the galaxy best?* Four methods now answer that on the same frozen lens, which makes the
comparison clean. Learning the lens jointly is a natural next step and deliberately out of scope.

**And why six lens numbers rather than more?** Ring size, radial slope, two for squashing, two for
external shear. The lens *centre* is pinned at the middle of the frame, because moving it is
equivalent to moving the galaxy — the two are degenerate, so freeing both would just make the fit
wander along a flat direction for no gain.

### Q10. "So the source box is just a grid, like the original work used?"

Yes, in the sense that matters: it is a regular square grid of pixels in the un-warped plane. Three
differences worth knowing:

1. **Size.** The original put a full-frame grid there — 128 or 254 pixels across. Ours is a small
   window, ±1.2 arcseconds at 48 by 48. Smaller window, far fewer unknowns, better determined.
2. **How it is read.** The original precomputed a big sparse matrix of overlap areas between the
   warped grid and the square grid, and applied it as a matrix multiply. We read the grid directly at
   the ray landing points by interpolation, live. Theirs conserves brightness exactly; ours is
   approximate but controlled, and ours lets the lens be a fitted parameter.
3. **What is on it.** In our better mode the grid does not hold the galaxy — it holds the *difference*
   between the galaxy and the smooth fit. That matters, because the smooth shape extends forever
   while a grid has an edge, so putting only the correction on the grid removes the edge problem
   entirely.

**And the edge problem is real, so mention it before they do:** a boxed grid has literally no
brightness outside the box. At ±0.8 arcseconds, 28% of the true galaxy's light falls outside — which
is most of why an earlier run came out too small and too peaked. At ±1.2 it is 11%. With the smooth
shape underneath, it stops mattering: the smooth part carries the outskirts, the grid only handles
the middle.

### Q11. "How is this different from the original DeepLense work?"

Same idea for the training signal — no sharp target, close the loop through the physics. Four
differences, and I would state them generously:

1. **The lens is fitted per image rather than fixed and circular with its size taken from the
   metadata.** Measured cost of the old choice: a round lens explains these arcs at 0.87 and only
   0.66 for the most squashed third, where a squashed lens with external shear reaches 0.997.
2. **The evaluation is against the true galaxy and the true physical parameters**, rather than
   image-quality scores — and we showed those image-quality scores are actively misleading here.
3. **The sharpness claim is tied to a measured magnification field and then tested by injection**,
   rather than being a chosen upscaling factor.
4. **B4 is their architecture on our lens**, so the last comparison is a controlled swap of exactly
   one component.

### Q12. "What would you do with another month?"

In order: fix the class sampling; use a real galaxy photograph as the source and show the free-form
part recovers what the smooth fit smooths away; use the second colour band; degrade the data to
ground-based conditions and check recovery survives — which is fully checkable because there is truth
on both sides; and attach an uncertainty to every prediction, which would turn the network's
under-estimation from a bias into a reported error bar.

---

# PART 4 — the fourteen numbers to have on the tip of your tongue

| number | what it is |
|---|---|
| **10.6–12.8 → 1.08** | reconstructed galaxy size, old pipeline → now |
| **0.987** | agreement between our reconstructed galaxy and the true one |
| **+0.96 / +0.91 / +0.76** | rank correlation for ring size / galaxy offset / lens squashing |
| **0.997 vs 0.87** | how well a squashed-plus-shear lens explains these arcs, versus a round one |
| **+0.10 vs +0.02** | what squashing buys, versus what refitting the ring size buys |
| **3.08×** | how much the lens stretches the galaxy — the ceiling on honest sharpening |
| **2.05× / 2.07×** | what our two networks actually ask for — inside that ceiling |
| **0.0017** | how much the agreement moves between 1× and 8× redrawing — i.e. flat |
| **0.63 pixels** | how well we recover the ring's position |
| **480 → 151** | model evaluations per fit, cold start → network start |
| **3302 → 3228** | fit score, cold → warm; better, not worse |
| **0.0408** | best free-form galaxy error (B4 with the smooth shape added) |
| **17–23%** | how much of an injected sub-pixel blob comes back |
| **+0.152 vs +0.056** | how strongly recovery correlates with photon count, versus with magnification |

---

# PART 5 — five things not to say

1. **Do not say "all three dark matter classes."** Everything is axion. Say so on slide 4 and move
   on.
2. **Do not say the free-form model beats the seven-parameter fit.** It does not on the scatter
   measures; it does on size and peak. Say exactly that.
3. **Do not present the magnification gate as working.** It came out 0.0414 against the control's
   0.0408. Present it as the fourth null and then give the explanation from slide 26 — that is a
   much better story than a marginal win would have been.
4. **Do not say "Levenberg–Marquardt" flatly.** Say "bounded least-squares fitting, using SciPy's
   trust-region-reflective solver."
5. **Do not claim magnification-adaptive smoothing helps.** Four attempts, four nulls. What you *can*
   claim, and it is stronger, is that you measured *why*.

---

# PART 6 — if you have five minutes instead of forty

Slides 2, 3, 10, 11, 18, 22, 26.

> "There is no sharp image to train on, so we train on physics. The old version kept the lens fixed
> and circular and got a galaxy eleven times too large — which is the *correct* response to a broken
> forward model, not a tuning failure. We now fit the lens per image and the galaxy comes out at
> 1.08. Because the answer is a formula rather than a picture, we can redraw it eight times finer with
> no loss of agreement. A network doing the same job in one pass is a poor estimator but an excellent
> starting point — it cuts the fit by two thirds and makes it slightly better. The newest model brings
> back this project's own original architecture on the corrected lens and halves the free-form error.
> And we can now demonstrate directly that structure smaller than a detector pixel comes back, at 17
> to 23 percent — though photon count predicts that three times more strongly than magnification does,
> which is why four attempts to exploit magnification came out flat."
