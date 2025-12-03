.. _changelog:

=========
CHANGELOG
=========


.. _changelog-v0.0.14:

v0.0.14 (2025-11-23)
====================

Bug Fixes
---------

* fix: chnage release (`f7a9655`_)

* fix: chnage release (`d78037d`_)

* fix: chnage release (`9ac3210`_)

* fix: chnage release (`30879c8`_)

* fix: fmax error correction (`c55574a`_)

* fix: demos (`5e1e794`_)

Unknown
-------

* add more demos (`cda7f2c`_)

* add more demos (`3902b28`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`9c9dcc7`_)

* add eeg plots (`9319193`_)

* add finance dataloader (`45d7467`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`9b5a329`_)

* Merge branch 'other_demos' into main (`49eb036`_)

* more hacking (`4f2e267`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD into main (`d0f7694`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD into main (`ca797e7`_)

* add vscode (`0498821`_)

* add improved multivar analysis (`cbe6beb`_)

* fix VI scaling (`f999333`_)

* fix VI init (`e5f023b`_)

* fix VI init (`c5e4af8`_)

* Merge pull request #20 from nz-gravity/allow_vi_standalone

allow VI standalone (`62f3dfc`_)

* allow VI standalone (`1a6574f`_)

* Merge pull request #19 from nz-gravity/extend-run_mcmc-with-only_vi-parameter

Add variational-only execution mode (`6814fb5`_)

* Add variational-only execution mode (`d3fcb33`_)

* fix scaling and one sided psd (`1837948`_)

* Merge pull request #17 from nz-gravity/testing_with_lisa

Add testing with LISA (`aa25677`_)

* add multivar study (`136aa51`_)

* add tests (`6c0003f`_)

* refactoring (`0ff8ee4`_)

* refactoring (`81c8485`_)

* Merge pull request #16 from nz-gravity/multivar_coarse_fixes

Add multivar coarse fix (`ab2ba5f`_)

* add fixes (`356b807`_)

* Add new knots

Coauthor: @pmat747 (`59d0dca`_)

* add better output (`dd92b85`_)

* Merge branch 'add_multivar_coarse_grain' (`5e73b7b`_)

* add lisa slurm (`791c0f4`_)

* Merge pull request #15 from nz-gravity/add_multivar_coarse_grain

add multivar coarse grain lnl (`9e2e2a7`_)

* add coarse grain lnl (`c3908e5`_)

* Merge pull request #14 from nz-gravity/add_multivar_averaged

add averaged data for multivar case (`500887f`_)

* fix conftext (`50e45a4`_)

* add change time blocks (`f9348bb`_)

* save quantiles directly instead of saving individaul PSD samples (`b6bba99`_)

* remove z matrix (`8b8d6e0`_)

* add averaged (`3c76b43`_)

* fix tests (`3455fb1`_)

* Merge pull request #11 from nz-gravity/coarse_lnl

Add Coarse-graining (univar) (`30f8ec1`_)

* add functional coarse lnl (`717a934`_)

* start working on coarse-LnL (`108ed22`_)

* Merge pull request #10 from nz-gravity/save_vi_diagnostics_before_sampling

save VI plots at the start (`e68d5de`_)

* save VI plots at the start (`d0cf75f`_)

* cleanup plotting (`4072420`_)

* Increase settings (`7709426`_)

* Refator plotting a bit to reuse code for VI (`cad0c7f`_)

* Add traceback (`fbcce61`_)

* Add welch PSD (`71252ef`_)

* autospectrum to PSD (`189e10d`_)

* Add gitbranch check (`0458967`_)

* Remove recomputation of ESS (`a464966`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`57f1bfc`_)

* hacking on the logger' (`ce7b24e`_)

* add few logs for VI init (`de0e2f8`_)

* fix PSD matrix --> real (`7550674`_)

* add extra debugs (`bc4386a`_)

* add logger (`335475c`_)

* fix logger (`43bc1ad`_)

* from print-->logger (`97da64f`_)

* Fix plotting error (`4502222`_)

* Add demo images (`0aaf092`_)

* Merge pull request #9 from nz-gravity/vi

Vi to init params (`4ed1781`_)

* init with VI (`78e0b7f`_)

* refactoring vi (`fd166d0`_)

* fix vi plotting (`eff19a7`_)

* add vi (`d5bd277`_)

* add more diagnostics for multivar case (`2ee6672`_)

* add blocked (`8331c33`_)

* add fixes for plotters (`d2f7274`_)

* Refactor NUTS samplers for shared blocks and log-phi reparam (`c4a261a`_)

* add txt file for faster simulation (`ec8e1a4`_)

* refactor pspline-sampling into its own reusable block and use einsum instead of @ (`b75ad1d`_)

* testing speedup attempts (`6503803`_)

* addd blocking (`31327de`_)

* hacking on lisa demo (`e708a6d`_)

* add plotting of results (`aa0e33c`_)

* reduce slurm requirements (`09dd1a2`_)

* add result extractor (`8b724c9`_)

* refactored multivar sampler so its a bit easeir to read (`6990664`_)

* hcking on lisa sim (`42d8b6e`_)

* Add sim study slurm (`729c050`_)

* fix sampler scaling (`299c03f`_)

* Merge pull request #7 from nz-gravity/nuts_lnz

add lnz computation for univar NUTS + MH (failing for multivar still) (`4e6cb0f`_)

* starting to work on multivar LnZ computation (`f364451`_)

* functioning morphZ lnZ computation (`f2df7ea`_)

* add lnz computation using MH posterior function (`4a80607`_)

* improve latex description (`65f6aba`_)

* add caching (`9f0c1b2`_)

* add GW tests (`765a0d9`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD

Also ran pre-commit formatter (`21d839a`_)

* add coverage check (`7b5cbba`_)

* Add some docs (`416a6c0`_)

* fix 'duplicate channel key' arviz error, and add ci_coverage functions (`d34c46a`_)

* Remove old multivar study (`b2ad55e`_)

* increase test size (`32ec7b6`_)

* Add coherence plotter (`2903276`_)

* Add simulation study for multivar (`f26033b`_)

* fix psd sccaling (`7386263`_)

* adding PSD plotting for multivar (`eb22d9a`_)

* add IAE (`efa0f8a`_)

* add a simple check for the 2d PSD nazeela (`4e1987f`_)

* Merge pull request #6 from nz-gravity/add_rescaling

add auto rescaling of data (`09e9609`_)

* add test (`5e47072`_)

* add auto rescaling of data (`d35f3de`_)

* load posterior-psd from idata (`5a054c5`_)

* convert pdgrm to numpy from jax (`5340e06`_)

* Allow plotting with saved PSD in idata (`c0cb019`_)

* Add different scaling for datasets (`5c1a99f`_)

* add plotting fixes (`bbeec72`_)

* remove dead function (`dbba00f`_)

* adjust 'slow' settings (`4f4cd5b`_)

* acceptance_rate to accept_prob (`9a34735`_)

* Add num-chains (`97d003c`_)

* Add extra-fields (`375c71c`_)

* Ave log-posterior (`c84160f`_)

* Add simulated dataset (`ec080a8`_)

* Testing improvements (`17dd4ec`_)

* remove dead code (`b437e54`_)

* use run_mcmc for the multivar dataset (`109032a`_)

* unified mcmc structure (`4c900ce`_)

* improve diagnostics (`29cd4f1`_)

* refactor to_arviz for better maintainability (`8c5e059`_)

* simplify to_arviz interface (`f6b0caf`_)

* remove sparcity comments (`ac9a555`_)

* create unified structure for creation of inference_data (`a5993c8`_)

* refactoring kwargs to run_mcmc (removed TypedDict of kwargs) (`51a716d`_)

* add idata saving (`e43851c`_)

* Batch spline eval (`4faf117`_)

* Use plotting function (`9b98ac6`_)

* Allow different plotting scales (`51504ca`_)

* refactor psd-matrix plotting (`d6f5118`_)

* some small speedups (`6367d31`_)

* Merge pull request #5 from nz-gravity/adding_new_sampler_base

Add multivar PSD estimator (`995697a`_)

* add multivar (`8fda37a`_)

* started adding new base sampler (`7a58aa1`_)

* refactor location for multivar code (`ce11966`_)

* refactor location of samplers (`0d1d24d`_)

* Add MultivariateLogPSplines class (`05b673b`_)

* Add MultivarFFT and MultivariateTimeseries (`ccf957f`_)

* refactor datatypes into new module (`549d711`_)

* Cant explicitly requested dtype <class 'jax.numpy.float64'>  -- users have to use JAX_ENABLE_X64 (`7b1b315`_)

* Add tqdm for reconstruction (`61c1b36`_)

* Add varma dataset (`0d11ec9`_)

* fixed workflow (`aa6b263`_)

* add multivar test (`7708a33`_)

* add multivar example (`83daeb8`_)

* add nsamples hack (`46d46c8`_)

* more hacking (`89106f8`_)

* add multivar PSD (`1328c06`_)

* more hacking on multivar PSD (`563f235`_)

* start working on multivar demo (`989e2cb`_)

* adding freq-grid for knot allocation (only knots at freq grid values) (`4b40c60`_)

* Merge pull request #4 from nz-gravity/add_morph_Lnz

add Morph-LnZ computation as an option (`d38b68d`_)

* add Morph-LnZ computation as an option (`3e26ded`_)

* add better docs (`c38e492`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`bd612cf`_)

.. _f7a9655: https://github.com/nz-gravity/LogPSplinePSD/commit/f7a965505ef429346033d7b5163a1eb132880eb8
.. _d78037d: https://github.com/nz-gravity/LogPSplinePSD/commit/d78037df1fecfe82a65f562860da04d14f546ed6
.. _9ac3210: https://github.com/nz-gravity/LogPSplinePSD/commit/9ac3210adc287028adef6cc2428be94fffdf1cde
.. _30879c8: https://github.com/nz-gravity/LogPSplinePSD/commit/30879c83971cad94351ed194c49e4cc88a552b66
.. _c55574a: https://github.com/nz-gravity/LogPSplinePSD/commit/c55574a330d97c2068b99071956ed92f7979d351
.. _5e1e794: https://github.com/nz-gravity/LogPSplinePSD/commit/5e1e79483bde9f21eaaf85e28b0ff6655c987c90
.. _cda7f2c: https://github.com/nz-gravity/LogPSplinePSD/commit/cda7f2c7ab5f845e66502dea1fc17c529117fd17
.. _3902b28: https://github.com/nz-gravity/LogPSplinePSD/commit/3902b280fec56995414230a5b04c16cfc2eea4b1
.. _9c9dcc7: https://github.com/nz-gravity/LogPSplinePSD/commit/9c9dcc73f6de28cd17a4b321744476b39fb641f1
.. _9319193: https://github.com/nz-gravity/LogPSplinePSD/commit/93191933ae10e84dca0de66bd02e4e1df8c71d7c
.. _45d7467: https://github.com/nz-gravity/LogPSplinePSD/commit/45d7467e7811616dd777a0d1c090cc097657f238
.. _9b5a329: https://github.com/nz-gravity/LogPSplinePSD/commit/9b5a329be4d999717f21f79b7b62122ce7e2f29d
.. _49eb036: https://github.com/nz-gravity/LogPSplinePSD/commit/49eb036af98a43928325474341a17812bbbd79c5
.. _4f2e267: https://github.com/nz-gravity/LogPSplinePSD/commit/4f2e267d1f9e163e647e722a28d9e01ba88ac5a6
.. _d0f7694: https://github.com/nz-gravity/LogPSplinePSD/commit/d0f7694fa784d105db52411ee399882d940b6022
.. _ca797e7: https://github.com/nz-gravity/LogPSplinePSD/commit/ca797e70d6aba22f4af4dd92c6bf73f068049dde
.. _0498821: https://github.com/nz-gravity/LogPSplinePSD/commit/049882181574c70576635cb6a46e34fe4caa5c23
.. _cbe6beb: https://github.com/nz-gravity/LogPSplinePSD/commit/cbe6beb8cbb9fd42086b0de2a0b6b3c261387771
.. _f999333: https://github.com/nz-gravity/LogPSplinePSD/commit/f999333d5e2d84906b31745a073cbe0f5d2002a2
.. _e5f023b: https://github.com/nz-gravity/LogPSplinePSD/commit/e5f023b6f8c1f164474060aed269ec502c1210b3
.. _c5e4af8: https://github.com/nz-gravity/LogPSplinePSD/commit/c5e4af814a62d525769ddd297dbbfcbcdd3c0562
.. _62f3dfc: https://github.com/nz-gravity/LogPSplinePSD/commit/62f3dfc2d10b0e7d8d44d384fe882991daa8856d
.. _1a6574f: https://github.com/nz-gravity/LogPSplinePSD/commit/1a6574f680954d97594e6353d7cd578494db6d9c
.. _6814fb5: https://github.com/nz-gravity/LogPSplinePSD/commit/6814fb5ee0faf3692a560701eaab9ea76b42b9d5
.. _d3fcb33: https://github.com/nz-gravity/LogPSplinePSD/commit/d3fcb33cf6f845f2a36b47190b9b885c126e2846
.. _1837948: https://github.com/nz-gravity/LogPSplinePSD/commit/183794833da60d1080ad0e2fc062383e106ad465
.. _aa25677: https://github.com/nz-gravity/LogPSplinePSD/commit/aa2567766a35897585baee653f7a1cc07249bae3
.. _136aa51: https://github.com/nz-gravity/LogPSplinePSD/commit/136aa51cac7589d62cab19fed6356615c6b46802
.. _6c0003f: https://github.com/nz-gravity/LogPSplinePSD/commit/6c0003f36d441637fd94cad067477a349796d85a
.. _0ff8ee4: https://github.com/nz-gravity/LogPSplinePSD/commit/0ff8ee4e67b736bc6cf2bf9ce5cd4093784f4cb2
.. _81c8485: https://github.com/nz-gravity/LogPSplinePSD/commit/81c84856c488e9e911f02da51e279df4081bfd8a
.. _ab2ba5f: https://github.com/nz-gravity/LogPSplinePSD/commit/ab2ba5ff84661b4c963a4f8e1a362bbc27677c76
.. _356b807: https://github.com/nz-gravity/LogPSplinePSD/commit/356b807e3dd57cf879be6c01812e73a8b861dc2c
.. _59d0dca: https://github.com/nz-gravity/LogPSplinePSD/commit/59d0dcaca55ed243f3dc53caecefe6db9ae05d38
.. _dd92b85: https://github.com/nz-gravity/LogPSplinePSD/commit/dd92b853783288635a7863d647733353d923ba20
.. _5e73b7b: https://github.com/nz-gravity/LogPSplinePSD/commit/5e73b7b8052154307e659259316e0b1ad0627341
.. _791c0f4: https://github.com/nz-gravity/LogPSplinePSD/commit/791c0f415b33138eb2a655915a9ffe26dbaed62d
.. _9e2e2a7: https://github.com/nz-gravity/LogPSplinePSD/commit/9e2e2a7084d0aef1c03e9783fab0763c4711d1d0
.. _c3908e5: https://github.com/nz-gravity/LogPSplinePSD/commit/c3908e5d1baa4041c9ed20e06058d008d23ceb7c
.. _500887f: https://github.com/nz-gravity/LogPSplinePSD/commit/500887f78e651f6b189cba3fb17684702b62fbe2
.. _50e45a4: https://github.com/nz-gravity/LogPSplinePSD/commit/50e45a4208044ab8f7b592d42a6cf2b788c1913b
.. _f9348bb: https://github.com/nz-gravity/LogPSplinePSD/commit/f9348bbff168ed482aaf1ee18d750ffa0614f7bc
.. _b6bba99: https://github.com/nz-gravity/LogPSplinePSD/commit/b6bba990ee29306da0b890e7973aea0018335035
.. _8b8d6e0: https://github.com/nz-gravity/LogPSplinePSD/commit/8b8d6e007c09e9def07b5a1074423cea01d689a9
.. _3c76b43: https://github.com/nz-gravity/LogPSplinePSD/commit/3c76b43b4085f94170e9111a0c4be826d5d0644d
.. _3455fb1: https://github.com/nz-gravity/LogPSplinePSD/commit/3455fb19bce60172120d5da955c5b10b22f172c3
.. _30f8ec1: https://github.com/nz-gravity/LogPSplinePSD/commit/30f8ec1a2a098d1eb8a4822b729a7c370c888ca0
.. _717a934: https://github.com/nz-gravity/LogPSplinePSD/commit/717a934a12d1dcfc84bf47506bcf168b8874d129
.. _108ed22: https://github.com/nz-gravity/LogPSplinePSD/commit/108ed228c854d28da6af24ae6e9ccce1e1d9222d
.. _e68d5de: https://github.com/nz-gravity/LogPSplinePSD/commit/e68d5de1d94f0fd225192b1cff4f38214c1013c7
.. _d0cf75f: https://github.com/nz-gravity/LogPSplinePSD/commit/d0cf75fca99d5f168ad0adea7117925730bc57ee
.. _4072420: https://github.com/nz-gravity/LogPSplinePSD/commit/40724200fba056b7fe48e3331817e59a20ea60a2
.. _7709426: https://github.com/nz-gravity/LogPSplinePSD/commit/7709426c9345e333d793561ee64f8c7e06e77d70
.. _cad0c7f: https://github.com/nz-gravity/LogPSplinePSD/commit/cad0c7f521e4b2c71bb366f35c7da4cd4aeddca9
.. _fbcce61: https://github.com/nz-gravity/LogPSplinePSD/commit/fbcce610eca28ab95232a31c0ea35516c4bf3e51
.. _71252ef: https://github.com/nz-gravity/LogPSplinePSD/commit/71252ef7cad534b1d8310fc90a13fe5c831d6310
.. _189e10d: https://github.com/nz-gravity/LogPSplinePSD/commit/189e10dcbd4996dc3ff3ef8b65f9171ea7fdfd71
.. _0458967: https://github.com/nz-gravity/LogPSplinePSD/commit/04589672219ee9bb4ff0a3d26db0fdac3af26e6e
.. _a464966: https://github.com/nz-gravity/LogPSplinePSD/commit/a4649660858ca538e7721204c87035c6f99a3536
.. _57f1bfc: https://github.com/nz-gravity/LogPSplinePSD/commit/57f1bfcb9aa74a9046174b8a8b3b3a1852345644
.. _ce7b24e: https://github.com/nz-gravity/LogPSplinePSD/commit/ce7b24e86b5fca9552429fdae96c51887564e5ed
.. _de0e2f8: https://github.com/nz-gravity/LogPSplinePSD/commit/de0e2f816182f27239063d85ac06f6a32c8b1e48
.. _7550674: https://github.com/nz-gravity/LogPSplinePSD/commit/755067456c45e8d6183077479733f86d715ba6ac
.. _bc4386a: https://github.com/nz-gravity/LogPSplinePSD/commit/bc4386ae9e477c9d76f65225d6398ada664cfa82
.. _335475c: https://github.com/nz-gravity/LogPSplinePSD/commit/335475cee9cdac1f6e3011edbba5d893b873d463
.. _43bc1ad: https://github.com/nz-gravity/LogPSplinePSD/commit/43bc1ad3732ed84bd7bddd99231fc09994362689
.. _97da64f: https://github.com/nz-gravity/LogPSplinePSD/commit/97da64f2ecc555ff86036dc6bb7f6d270c4690be
.. _4502222: https://github.com/nz-gravity/LogPSplinePSD/commit/45022229adf67fa753027ebe00052a56102c062b
.. _0aaf092: https://github.com/nz-gravity/LogPSplinePSD/commit/0aaf09257feff50daeac50a3b0fada5897000835
.. _4ed1781: https://github.com/nz-gravity/LogPSplinePSD/commit/4ed1781665849e344d484eb0fa1cc57b89b536ec
.. _78e0b7f: https://github.com/nz-gravity/LogPSplinePSD/commit/78e0b7ff0b1aee66415472baaea91ac76d998f4a
.. _fd166d0: https://github.com/nz-gravity/LogPSplinePSD/commit/fd166d002af0167db102d9e6bc737c58b8f42edf
.. _eff19a7: https://github.com/nz-gravity/LogPSplinePSD/commit/eff19a79dc9042094de1b68b192777b71cb50fd7
.. _d5bd277: https://github.com/nz-gravity/LogPSplinePSD/commit/d5bd2770dd2811bb5e90892c905a4cd55b372e01
.. _2ee6672: https://github.com/nz-gravity/LogPSplinePSD/commit/2ee6672c9c63f7d26ad1273e378a93e982034393
.. _8331c33: https://github.com/nz-gravity/LogPSplinePSD/commit/8331c330b0007a1b7463dcdf1b11fd342f9ec5bf
.. _d2f7274: https://github.com/nz-gravity/LogPSplinePSD/commit/d2f7274a0325f18e7c7ed344e4e2af4c2d6c660f
.. _c4a261a: https://github.com/nz-gravity/LogPSplinePSD/commit/c4a261a5cc4901783ca3894a722494fc8385cfc1
.. _ec8e1a4: https://github.com/nz-gravity/LogPSplinePSD/commit/ec8e1a49a6302180f0e39b3d02ffeaf89303e744
.. _b75ad1d: https://github.com/nz-gravity/LogPSplinePSD/commit/b75ad1d7ec3da241a178705937c782d72644d763
.. _6503803: https://github.com/nz-gravity/LogPSplinePSD/commit/65038039a9e5f8bccd160b8981d21423e0a73f72
.. _31327de: https://github.com/nz-gravity/LogPSplinePSD/commit/31327de1d3cd3f609bc6e6d6fbced9b03bfb4e91
.. _e708a6d: https://github.com/nz-gravity/LogPSplinePSD/commit/e708a6d15e4166d001b27e8477275f216457e722
.. _aa0e33c: https://github.com/nz-gravity/LogPSplinePSD/commit/aa0e33c9b9cc5cbee58729cb13fb0cc9dfabc978
.. _09dd1a2: https://github.com/nz-gravity/LogPSplinePSD/commit/09dd1a2e9eb27660151f2f6d15cf35fbedd5ecdc
.. _8b724c9: https://github.com/nz-gravity/LogPSplinePSD/commit/8b724c98dc7426f944ed90227050bc8fed5b1ad8
.. _6990664: https://github.com/nz-gravity/LogPSplinePSD/commit/6990664eaa4967fe00e10fd039f98f3d68020636
.. _42d8b6e: https://github.com/nz-gravity/LogPSplinePSD/commit/42d8b6ec12c1d4c1aa43a600acbc36ccc167b59a
.. _729c050: https://github.com/nz-gravity/LogPSplinePSD/commit/729c0509af833a58934b9e53e920bc414f86648f
.. _299c03f: https://github.com/nz-gravity/LogPSplinePSD/commit/299c03fee58c2fcdf98c18111b13597b64e83f2e
.. _4e6cb0f: https://github.com/nz-gravity/LogPSplinePSD/commit/4e6cb0fe43a75d4b170cca7d99fafaadd1d5aabf
.. _f364451: https://github.com/nz-gravity/LogPSplinePSD/commit/f3644519ac64cc9098000a2b3ee76fef7dd29cae
.. _f2df7ea: https://github.com/nz-gravity/LogPSplinePSD/commit/f2df7eaeebf143b678b0773ca0bebe58f5f2cdf8
.. _4a80607: https://github.com/nz-gravity/LogPSplinePSD/commit/4a80607078f4797c46a6e72934c019a78a4d5b3e
.. _65f6aba: https://github.com/nz-gravity/LogPSplinePSD/commit/65f6aba4cd168b2ae13211be9e58d1dfd765c5a2
.. _9f0c1b2: https://github.com/nz-gravity/LogPSplinePSD/commit/9f0c1b28a3f56ce0617f70c391cc92101e7af198
.. _765a0d9: https://github.com/nz-gravity/LogPSplinePSD/commit/765a0d98a1392804670f1974af8b226ca604bed5
.. _21d839a: https://github.com/nz-gravity/LogPSplinePSD/commit/21d839a5d97e4075725129b285691b949f22a9be
.. _7b5cbba: https://github.com/nz-gravity/LogPSplinePSD/commit/7b5cbba6c4401bd8b68ac25cef176da9b5712083
.. _416a6c0: https://github.com/nz-gravity/LogPSplinePSD/commit/416a6c0882572ed44b2ef0015793c21500c19044
.. _d34c46a: https://github.com/nz-gravity/LogPSplinePSD/commit/d34c46aeca38f66b225b256ef4fd60d18dd6753e
.. _b2ad55e: https://github.com/nz-gravity/LogPSplinePSD/commit/b2ad55e646b1a0a6d8247970c4a705234f3d470b
.. _32ec7b6: https://github.com/nz-gravity/LogPSplinePSD/commit/32ec7b6fdc5f7c47f834aa42dd76f50a77e8dbf2
.. _2903276: https://github.com/nz-gravity/LogPSplinePSD/commit/29032762266538c96013bc1c6247de0cc17c0a83
.. _f26033b: https://github.com/nz-gravity/LogPSplinePSD/commit/f26033b074620dfc7adc6d962ff44e3ae035fabb
.. _7386263: https://github.com/nz-gravity/LogPSplinePSD/commit/73862630589aa6debe0d2fdd9052fd0995230e8d
.. _eb22d9a: https://github.com/nz-gravity/LogPSplinePSD/commit/eb22d9a8131e2fb3e1afc2060060a5adc937ee97
.. _efa0f8a: https://github.com/nz-gravity/LogPSplinePSD/commit/efa0f8a928c0658279833bdc8032f1aff562b9e9
.. _4e1987f: https://github.com/nz-gravity/LogPSplinePSD/commit/4e1987f0d44263be6f2ce6fbe5e45203894eea6f
.. _09e9609: https://github.com/nz-gravity/LogPSplinePSD/commit/09e9609f6b2d10630ff9440331d549f8901f24d4
.. _5e47072: https://github.com/nz-gravity/LogPSplinePSD/commit/5e47072139bc8dbd589dbac820487449e465d6b6
.. _d35f3de: https://github.com/nz-gravity/LogPSplinePSD/commit/d35f3de1f728035bee247938110dbeb3e87913eb
.. _5a054c5: https://github.com/nz-gravity/LogPSplinePSD/commit/5a054c52b7d39b53c50d126643dda99e775ee1e6
.. _5340e06: https://github.com/nz-gravity/LogPSplinePSD/commit/5340e06e82b7096d9e43881f32ab234b3a29e802
.. _c0cb019: https://github.com/nz-gravity/LogPSplinePSD/commit/c0cb019017e90a0cd6beffb26d2f54e00b13de96
.. _5c1a99f: https://github.com/nz-gravity/LogPSplinePSD/commit/5c1a99f583d59b3533bb763d7088d02eb7182ef7
.. _bbeec72: https://github.com/nz-gravity/LogPSplinePSD/commit/bbeec72bd58613a32b4c33bf7ac9e3eafdb31072
.. _dbba00f: https://github.com/nz-gravity/LogPSplinePSD/commit/dbba00f6aa98bd13c9febfa8fc2dd33ec5460d1f
.. _4f4cd5b: https://github.com/nz-gravity/LogPSplinePSD/commit/4f4cd5b5b7089ff6515ef4c22aaaddbf77301a83
.. _9a34735: https://github.com/nz-gravity/LogPSplinePSD/commit/9a34735133a8a908b53eb5a90831280f41acae7c
.. _97d003c: https://github.com/nz-gravity/LogPSplinePSD/commit/97d003c45245ad321f05b79be7a1d181592306b0
.. _375c71c: https://github.com/nz-gravity/LogPSplinePSD/commit/375c71ce10c4387fec2ebcce0ce51b56d0398047
.. _c84160f: https://github.com/nz-gravity/LogPSplinePSD/commit/c84160f8e9d2d219a79c887646819994e37a5985
.. _ec080a8: https://github.com/nz-gravity/LogPSplinePSD/commit/ec080a8a8afffcb12e8416fabc0e4d2af3c379e2
.. _17dd4ec: https://github.com/nz-gravity/LogPSplinePSD/commit/17dd4ec3d60f8362fc26c0655abb008aa3fa4a30
.. _b437e54: https://github.com/nz-gravity/LogPSplinePSD/commit/b437e5440691dcb1263bd5eada4b1d1d67f23b85
.. _109032a: https://github.com/nz-gravity/LogPSplinePSD/commit/109032a02da599a2dfd5e684735282307661273e
.. _4c900ce: https://github.com/nz-gravity/LogPSplinePSD/commit/4c900cebf348851993d3a8110fba657c39213e65
.. _29cd4f1: https://github.com/nz-gravity/LogPSplinePSD/commit/29cd4f1cdb712c9ed23bce5d5f66f2fa07be374f
.. _8c5e059: https://github.com/nz-gravity/LogPSplinePSD/commit/8c5e059b64fd70af43a0eb1a3d7bc99cccf78182
.. _f6b0caf: https://github.com/nz-gravity/LogPSplinePSD/commit/f6b0caf8f114c11690b633033556372ca468a7fe
.. _ac9a555: https://github.com/nz-gravity/LogPSplinePSD/commit/ac9a5555c64090806b52067a2b275d61bc354789
.. _a5993c8: https://github.com/nz-gravity/LogPSplinePSD/commit/a5993c8b872dad5b528a1c7f5a5191f047365360
.. _51a716d: https://github.com/nz-gravity/LogPSplinePSD/commit/51a716dc68e0caf89294d84c6a3b9b72755d7bfc
.. _e43851c: https://github.com/nz-gravity/LogPSplinePSD/commit/e43851cd0cbdb80c7b37eb9cb03ad936b3e22fca
.. _4faf117: https://github.com/nz-gravity/LogPSplinePSD/commit/4faf11728c363da50402e4fa5293c5e7644b42b5
.. _9b98ac6: https://github.com/nz-gravity/LogPSplinePSD/commit/9b98ac6851e81fc34de471b90eff74b9258ed544
.. _51504ca: https://github.com/nz-gravity/LogPSplinePSD/commit/51504ca650c11adaf464bfaf98a6acd1f089620f
.. _d6f5118: https://github.com/nz-gravity/LogPSplinePSD/commit/d6f5118edbcb202b5f79193d82fddeba0f8d6422
.. _6367d31: https://github.com/nz-gravity/LogPSplinePSD/commit/6367d316cfea7f8638fb83e35390564858c1f41d
.. _995697a: https://github.com/nz-gravity/LogPSplinePSD/commit/995697a679235d196e846c88199bca28bf8f17a6
.. _8fda37a: https://github.com/nz-gravity/LogPSplinePSD/commit/8fda37aa93f0ac92b0b9fd00f9f5e08ffdce1bdf
.. _7a58aa1: https://github.com/nz-gravity/LogPSplinePSD/commit/7a58aa19343f934a5cf6739fb75ad63d60ce41f7
.. _ce11966: https://github.com/nz-gravity/LogPSplinePSD/commit/ce11966f57acb958fbeba66ed4e03d006005db12
.. _0d1d24d: https://github.com/nz-gravity/LogPSplinePSD/commit/0d1d24d12dc5b54ea92b3cb82771facd7ce9acae
.. _05b673b: https://github.com/nz-gravity/LogPSplinePSD/commit/05b673b8e3945e9badf3972e034e85e94dfd3aee
.. _ccf957f: https://github.com/nz-gravity/LogPSplinePSD/commit/ccf957fccac9073b982fa30260ee5b2d6191ef6a
.. _549d711: https://github.com/nz-gravity/LogPSplinePSD/commit/549d7118a6535fa229e51ee469d7b47a9c29cdb2
.. _7b1b315: https://github.com/nz-gravity/LogPSplinePSD/commit/7b1b315165bc373857df98133a1b0111ece76e07
.. _61c1b36: https://github.com/nz-gravity/LogPSplinePSD/commit/61c1b36b6741830f6d513c3942d7d6b76192c60d
.. _0d11ec9: https://github.com/nz-gravity/LogPSplinePSD/commit/0d11ec98fa70e9e180449cc19bb1b2c62e38d44c
.. _aa6b263: https://github.com/nz-gravity/LogPSplinePSD/commit/aa6b2637c67b939310d473958f494b6cccb3b8c4
.. _7708a33: https://github.com/nz-gravity/LogPSplinePSD/commit/7708a33aef5d96d15b596d13384471137c1de95d
.. _83daeb8: https://github.com/nz-gravity/LogPSplinePSD/commit/83daeb89831cd470b7be3b5a6084ca20fd54469e
.. _46d46c8: https://github.com/nz-gravity/LogPSplinePSD/commit/46d46c825de2641caace36248da5f64ce1e87d1f
.. _89106f8: https://github.com/nz-gravity/LogPSplinePSD/commit/89106f82cf016d175b81cab835ae108e89ee3019
.. _1328c06: https://github.com/nz-gravity/LogPSplinePSD/commit/1328c06306a1272d3dc680ffe5b6d7f7e0b4d299
.. _563f235: https://github.com/nz-gravity/LogPSplinePSD/commit/563f235529b5b9a129201bb6c5dc916ecc11a303
.. _989e2cb: https://github.com/nz-gravity/LogPSplinePSD/commit/989e2cb47a670a0bdb4552e89383e10c29972731
.. _4b40c60: https://github.com/nz-gravity/LogPSplinePSD/commit/4b40c6008baf470e8a7598e22aee037085f2f9ae
.. _d38b68d: https://github.com/nz-gravity/LogPSplinePSD/commit/d38b68dd1c653d32c523d64b942ef61f51f5f4a7
.. _3e26ded: https://github.com/nz-gravity/LogPSplinePSD/commit/3e26ded63bd0a17bf52d87acdbb67e29c65df064
.. _c38e492: https://github.com/nz-gravity/LogPSplinePSD/commit/c38e49264c55bc365973d00e1d0722dd46bd8099
.. _bd612cf: https://github.com/nz-gravity/LogPSplinePSD/commit/bd612cfdf3a1f2d195cb03927c4ed17ad3f5ce8f


.. _changelog-v0.0.13:

v0.0.13 (2025-09-11)
====================

Bug Fixes
---------

* fix: plotly plotting (`dca5de4`_)

Chores
------

* chore(release): 0.0.13 (`08d8c2f`_)

Unknown
-------

* Add patricio's knot allocation

Co-authored-by: Patricio Maturana-Russel <pmat747@users.noreply.github.com> (`7022e96`_)

* Improve type hints (`e6b7025`_)

* Add patricio's knot allocation

Co-authored-by: Patricio Maturana-Russel <pmat747@users.noreply.github.com> (`9341b76`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD into main (`0700b40`_)

.. _dca5de4: https://github.com/nz-gravity/LogPSplinePSD/commit/dca5de44934547331f0b20bb9ad2101b2a180d96
.. _08d8c2f: https://github.com/nz-gravity/LogPSplinePSD/commit/08d8c2f2b172bf6ae830d3ee02bb3afde9327ad5
.. _7022e96: https://github.com/nz-gravity/LogPSplinePSD/commit/7022e9642e89c8c73d1bc1f78d9adafece5ebee4
.. _e6b7025: https://github.com/nz-gravity/LogPSplinePSD/commit/e6b7025f55d00a2eb4d3cbeed8e559ae7448b1fc
.. _9341b76: https://github.com/nz-gravity/LogPSplinePSD/commit/9341b7654c37fd682dbc0be296af5562e4c06852
.. _0700b40: https://github.com/nz-gravity/LogPSplinePSD/commit/0700b4089b7e472bc9be8ac88f1fe1c47a90f095


.. _changelog-v0.0.12:

v0.0.12 (2025-09-10)
====================

Bug Fixes
---------

* fix: adjust default settings (`9123429`_)

* fix: made plotting changes (`4e110d0`_)

Chores
------

* chore(release): 0.0.12 (`de17f2b`_)

Documentation
-------------

* docs: add plots (`d91ff0a`_)

* docs: add colab button (`786a454`_)

* docs: add some notes for commits (`d0f834f`_)

Unknown
-------

* fix conflicts (`6eb09f0`_)

* qol changes (`83cdaa4`_)

* add simulation study (`19da3bd`_)

* testing with higher diff matrix order (`d27c5ff`_)

* Add info on penalty matrix in repr (`20971e0`_)

* Add simulation study files (`be5df30`_)

* Remove jax version fix (`e0f337e`_)

* Allow for higher penalty matrix (`84734b1`_)

* Print the model init (`c47865b`_)

* Add a script to explor N-knots vs the IAE (`3311fcc`_)

* Specify dtype for penalty matrix (`6b65536`_)

* Load posterior PSD from arviz, allow passing of path to inferenec objec (`0bd7114`_)

* Add units (`f8bbf21`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`6f87c63`_)

.. _9123429: https://github.com/nz-gravity/LogPSplinePSD/commit/9123429b2eeb98d3338b6cfbe210e25114ce0e52
.. _4e110d0: https://github.com/nz-gravity/LogPSplinePSD/commit/4e110d06660a8624f7da3304e09045536c341eb1
.. _de17f2b: https://github.com/nz-gravity/LogPSplinePSD/commit/de17f2bd5f6e6d577e3b5517a3dbf07a16703bfe
.. _d91ff0a: https://github.com/nz-gravity/LogPSplinePSD/commit/d91ff0aebc229c214df373d14ce31e21d38fb70f
.. _786a454: https://github.com/nz-gravity/LogPSplinePSD/commit/786a454a6fa652f7565b1afe65b7d21c43b2c773
.. _d0f834f: https://github.com/nz-gravity/LogPSplinePSD/commit/d0f834f046821813c7ca62a88ec2c8038ebb167e
.. _6eb09f0: https://github.com/nz-gravity/LogPSplinePSD/commit/6eb09f06bb28e4c74a4f9c5b694afc93093599f2
.. _83cdaa4: https://github.com/nz-gravity/LogPSplinePSD/commit/83cdaa44696d18430d7f7e2371f302c32ca18b66
.. _19da3bd: https://github.com/nz-gravity/LogPSplinePSD/commit/19da3bd99aae4337c47fecf23b2c4132cb0e0693
.. _d27c5ff: https://github.com/nz-gravity/LogPSplinePSD/commit/d27c5ff1c636df2974463d4e599370a0bd4a09f0
.. _20971e0: https://github.com/nz-gravity/LogPSplinePSD/commit/20971e049f065db0a8af0fffc66a4d315205184e
.. _be5df30: https://github.com/nz-gravity/LogPSplinePSD/commit/be5df3049a3895be881618d9386a1f2885696149
.. _e0f337e: https://github.com/nz-gravity/LogPSplinePSD/commit/e0f337ed8ecc1abccdd2d364158f864b69090a8d
.. _84734b1: https://github.com/nz-gravity/LogPSplinePSD/commit/84734b1e33cc58d2be31a4ee4e22d2bafc67c03f
.. _c47865b: https://github.com/nz-gravity/LogPSplinePSD/commit/c47865b01a9223024024624cb35375168ab65bfd
.. _3311fcc: https://github.com/nz-gravity/LogPSplinePSD/commit/3311fccf5006d809857e335bfb4aab46a88c3a56
.. _6b65536: https://github.com/nz-gravity/LogPSplinePSD/commit/6b65536fa944c6418b6adce144cf6ac0fb3eb738
.. _0bd7114: https://github.com/nz-gravity/LogPSplinePSD/commit/0bd7114a4d043e7ebd1393c4df376360617d1128
.. _f8bbf21: https://github.com/nz-gravity/LogPSplinePSD/commit/f8bbf213d7f220f618544175b9db7f9276b3ab5a
.. _6f87c63: https://github.com/nz-gravity/LogPSplinePSD/commit/6f87c638c0d5d5c4e20369c8354bf359796c7489


.. _changelog-v0.0.11:

v0.0.11 (2025-09-02)
====================

Bug Fixes
---------

* fix: refactor line-locator name to lvk-allocator (`35f4c30`_)

Chores
------

* chore(release): 0.0.11 (`156d8ff`_)

Continuous Integration
----------------------

* ci: update pypi to only release for specific tags (`21cf551`_)

Unknown
-------

* run precommits (`ca62dfb`_)

* hacking on lvk knot loc (`c53b3da`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`97bd97b`_)

.. _35f4c30: https://github.com/nz-gravity/LogPSplinePSD/commit/35f4c307d98c2c6406fd155a17655c776ce25a43
.. _156d8ff: https://github.com/nz-gravity/LogPSplinePSD/commit/156d8ff8bf6567829760c1e55ddd8ff13eaaff76
.. _21cf551: https://github.com/nz-gravity/LogPSplinePSD/commit/21cf551f7c2a986932eaa3081114b39e5a438776
.. _ca62dfb: https://github.com/nz-gravity/LogPSplinePSD/commit/ca62dfb889af5cb20fe938a7046bf4c1c4aa522a
.. _c53b3da: https://github.com/nz-gravity/LogPSplinePSD/commit/c53b3da594acd8bc8e53a934cbf7ac72effd8a6b
.. _97bd97b: https://github.com/nz-gravity/LogPSplinePSD/commit/97bd97b55c28a3683e5674c428ea0e26e8e9f74a


.. _changelog-v0.0.10:

v0.0.10 (2025-08-28)
====================

Bug Fixes
---------

* fix: LVK knot allocation fix (`6ffa886`_)

Chores
------

* chore(release): 0.0.10 (`82bc78c`_)

Unknown
-------

* pypi onl after pytest passes (`48b923f`_)

.. _6ffa886: https://github.com/nz-gravity/LogPSplinePSD/commit/6ffa88680cfc2b297a99eaba87c1a471df668af5
.. _82bc78c: https://github.com/nz-gravity/LogPSplinePSD/commit/82bc78c6a4feab3eeb5cc31d44ebb93fcf1e4a13
.. _48b923f: https://github.com/nz-gravity/LogPSplinePSD/commit/48b923f1ed813cc47b980a8af9a6f6a201c74be4


.. _changelog-v0.0.9:

v0.0.9 (2025-08-25)
===================

Bug Fixes
---------

* fix: pytests (`435ca66`_)

Chores
------

* chore(release): 0.0.9 (`a94ef60`_)

Unknown
-------

* add LVK allocation (`9f93242`_)

* add more tests for PSD diagnostics (`55a94d3`_)

* add: add LVK code testing (`dfa0c3d`_)

* add basis comparison (`085bc80`_)

* add cut (`85909ab`_)

* refactor docs to work with new API (`ee05bca`_)

* refactor preprocessing (`01790f9`_)

.. _435ca66: https://github.com/nz-gravity/LogPSplinePSD/commit/435ca666329345feb295eb23a16b962fb57120e0
.. _a94ef60: https://github.com/nz-gravity/LogPSplinePSD/commit/a94ef60984fe1f441f884563498288d1fbf0669f
.. _9f93242: https://github.com/nz-gravity/LogPSplinePSD/commit/9f932424ab62aae038cb99ab770e2488596a648a
.. _55a94d3: https://github.com/nz-gravity/LogPSplinePSD/commit/55a94d39a9c928398ae7c0995ccd54ba7de88838
.. _dfa0c3d: https://github.com/nz-gravity/LogPSplinePSD/commit/dfa0c3d535b54a752099a35fe21b515422e9d08c
.. _085bc80: https://github.com/nz-gravity/LogPSplinePSD/commit/085bc80c4718b722b4aa487fa980f586b790db7b
.. _85909ab: https://github.com/nz-gravity/LogPSplinePSD/commit/85909ab2869e3fe9f62f97776b7089cc6ab1ee66
.. _ee05bca: https://github.com/nz-gravity/LogPSplinePSD/commit/ee05bca1ecad4755855c16ed345ca9d6f2a010bd
.. _01790f9: https://github.com/nz-gravity/LogPSplinePSD/commit/01790f98edbe7d4905640efb9b7fb28e55c29f87


.. _changelog-v0.0.8:

v0.0.8 (2025-07-24)
===================

Bug Fixes
---------

* fix: refactoring (`cc87bfd`_)

Chores
------

* chore(release): 0.0.8 (`f32b903`_)

Unknown
-------

* add SVI testing (`09bc512`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`26d4210`_)

.. _cc87bfd: https://github.com/nz-gravity/LogPSplinePSD/commit/cc87bfdf4a90e38f190bcf2b5a01a0c04ae53baa
.. _f32b903: https://github.com/nz-gravity/LogPSplinePSD/commit/f32b90326fb81c231fc48db66b69828707113cd9
.. _09bc512: https://github.com/nz-gravity/LogPSplinePSD/commit/09bc5126823cbc28b2543c95bb76e01d7ef630b2
.. _26d4210: https://github.com/nz-gravity/LogPSplinePSD/commit/26d4210495d5ab1b8367dd75d506cb5690ad752f


.. _changelog-v0.0.7:

v0.0.7 (2025-07-21)
===================

Chores
------

* chore(release): 0.0.7 (`3661bf5`_)

Unknown
-------

* fix typo (`bdaa71f`_)

* t push
Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`06db40f`_)

.. _3661bf5: https://github.com/nz-gravity/LogPSplinePSD/commit/3661bf5da22ac4a87939910d481e81e9cac736fb
.. _bdaa71f: https://github.com/nz-gravity/LogPSplinePSD/commit/bdaa71f4be416c7dd1a354d13c6267f64062c3ac
.. _06db40f: https://github.com/nz-gravity/LogPSplinePSD/commit/06db40f2358de0081bf8845dabbcc6552882e09c


.. _changelog-v0.0.6:

v0.0.6 (2025-07-21)
===================

Bug Fixes
---------

* fix: add sampler option (`09ce185`_)

* fix: add RNG logging and verbosity (`e2cb737`_)

Chores
------

* chore(release): 0.0.6 (`2190d9a`_)

Unknown
-------

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`a69cd29`_)

.. _09ce185: https://github.com/nz-gravity/LogPSplinePSD/commit/09ce18588a0c7100fb55d1133bfd843c46f6b17f
.. _e2cb737: https://github.com/nz-gravity/LogPSplinePSD/commit/e2cb7372ba51127727d7598f6c1dcad7bf038449
.. _2190d9a: https://github.com/nz-gravity/LogPSplinePSD/commit/2190d9a7dc255c4740608364d389f7fcceafb801
.. _a69cd29: https://github.com/nz-gravity/LogPSplinePSD/commit/a69cd29df0326f764176b1ef586a270b7f6b7d2c


.. _changelog-v0.0.5:

v0.0.5 (2025-07-21)
===================

Bug Fixes
---------

* fix: update runner (`85edc41`_)

* fix: add benchmarking fix for cli (`8914c67`_)

Chores
------

* chore(release): 0.0.5 (`f3cfd37`_)

Unknown
-------

* add benchmarking code (`1361278`_)

* add ESS comparison (`88b8792`_)

.. _85edc41: https://github.com/nz-gravity/LogPSplinePSD/commit/85edc41f866cfc9200c7267cceaae2a0c681fd82
.. _8914c67: https://github.com/nz-gravity/LogPSplinePSD/commit/8914c6733dbcecd1543cde23f20553ced1a6fbba
.. _f3cfd37: https://github.com/nz-gravity/LogPSplinePSD/commit/f3cfd3750f940f1c12740aa5fe82c7c05384df21
.. _1361278: https://github.com/nz-gravity/LogPSplinePSD/commit/1361278de8c80c9e2509480325f7f160bf833259
.. _88b8792: https://github.com/nz-gravity/LogPSplinePSD/commit/88b879285577f13e53c844f19f18c26cb8cd4cb5


.. _changelog-v0.0.4:

v0.0.4 (2025-07-17)
===================

Bug Fixes
---------

* fix: update inference result saving/loading (`c1c6273`_)

Chores
------

* chore(release): 0.0.4 (`b4016a2`_)

Unknown
-------

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`49e3a72`_)

.. _c1c6273: https://github.com/nz-gravity/LogPSplinePSD/commit/c1c627301a886a792c25b60fa85dee13d173eceb
.. _b4016a2: https://github.com/nz-gravity/LogPSplinePSD/commit/b4016a25e0e8ae3fa6d614cc442d36e53bfe335c
.. _49e3a72: https://github.com/nz-gravity/LogPSplinePSD/commit/49e3a727d479206fa16eeba3b8828acb48141356


.. _changelog-v0.0.3:

v0.0.3 (2025-07-16)
===================

Chores
------

* chore(release): 0.0.3 (`4ef53a9`_)

Unknown
-------

* update readme links (`0385b74`_)

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`4c72cc2`_)

.. _4ef53a9: https://github.com/nz-gravity/LogPSplinePSD/commit/4ef53a986e41573a8b159416f0ce127aeb202872
.. _0385b74: https://github.com/nz-gravity/LogPSplinePSD/commit/0385b745795411e7e42790da58269c43ff5611d5
.. _4c72cc2: https://github.com/nz-gravity/LogPSplinePSD/commit/4c72cc2028d58dceeb717915f6bf2d9fb194a9c2


.. _changelog-v0.0.2:

v0.0.2 (2025-07-16)
===================

Bug Fixes
---------

* fix: fix pypi name bug (`b4b06db`_)

* fix: pypi readme fix and updating demo (`7ecc602`_)

Chores
------

* chore(release): 0.0.2 (`1d35bb9`_)

Unknown
-------

* Merge branch 'main' of github.com:nz-gravity/LogPSplinePSD (`5121194`_)

.. _b4b06db: https://github.com/nz-gravity/LogPSplinePSD/commit/b4b06db36c36e72793d659e317ce26af52108865
.. _7ecc602: https://github.com/nz-gravity/LogPSplinePSD/commit/7ecc602bc7c066bcd9b86be4340575d10057c01c
.. _1d35bb9: https://github.com/nz-gravity/LogPSplinePSD/commit/1d35bb982f74f1ae9be5021a983f4267b0627cfc
.. _5121194: https://github.com/nz-gravity/LogPSplinePSD/commit/5121194a38c18dfbf31e7bcc9c3751409d4cb9b7


.. _changelog-v0.0.1:

v0.0.1 (2025-07-16)
===================

Bug Fixes
---------

* fix: add gwpy for dev options (`964e40e`_)

* fix: add arviz (`7a0925c`_)

* fix: add diagostics and ar dataset for tstig (`a43ee40`_)

* fix: add demo to docs (`235c3ec`_)

* fix: init weights with mse istead of lnl (`9df1e5d`_)

Chores
------

* chore(release): 0.0.1 (`027591f`_)

Unknown
-------

* Update pypi.yml (`88c8f9b`_)

* edit readme (`60af98d`_)

* add: add option for mh and nuts (`3a08b99`_)

* refactoring to use a common parent class (`1fb79e8`_)

* change to just vanilla metropolis-hastings (get rid of covar matrix adaptation) (`b0cd698`_)

* Merge pull request #3 from nz-gravity/adding_adaptive_mcmc

Adding adaptive MCMC (`fd9a95b`_)

* init (`c41038c`_)

* fix tests (`328e854`_)

* Update docs.yml (`5877ec0`_)

* Update README.rst (`20d3f39`_)

* add line locator (`dc6469c`_)

* add fix (`7f32bbb`_)

* refactor (`a061028`_)

* add docs (`6b3905f`_)

* add examples (`cf42e6f`_)

* add psd approx (`18d0075`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD (`a2035bb`_)

* Create LICENSE (`8fff25b`_)

* fix readme (`939cbdb`_)

* add workflows (`73fd427`_)

* Merge branch 'main' of github.com:avivajpeyi/LogPSplinePSD (`194fae8`_)

* Merge pull request #1 from avivajpeyi/pre-commit-ci-update-config

[pre-commit.ci] pre-commit autoupdate (`7231c3b`_)

* [pre-commit.ci] auto fixes from pre-commit.com hooks

for more information, see https://pre-commit.ci (`6641a63`_)

* [pre-commit.ci] pre-commit autoupdate

updates:
- [github.com/pre-commit/pre-commit-hooks: v4.5.0 → v5.0.0](https://github.com/pre-commit/pre-commit-hooks/compare/v4.5.0...v5.0.0)
- https://github.com/pre-commit/mirrors-isort → https://github.com/PyCQA/isort
- [github.com/PyCQA/isort: v5.10.1 → 6.0.1](https://github.com/PyCQA/isort/compare/v5.10.1...6.0.1)
- https://github.com/ambv/black → https://github.com/psf/black
- [github.com/psf/black: 23.10.0 → 25.1.0](https://github.com/psf/black/compare/23.10.0...25.1.0)
- [github.com/psf/black: 23.10.0 → 25.1.0](https://github.com/psf/black/compare/23.10.0...25.1.0) (`98ae77a`_)

* add welch psd (`d7121d6`_)

* add LVK plots (`f818caa`_)

* add LVK example and parametric model (`0666415`_)

* hackig on alternative model (`4197563`_)

* add LVK example (`922f870`_)

* add LVK example (`4944aa1`_)

* add lvk noise (`d93f36b`_)

* add tests (`c9e3c79`_)

* more hacking (`fda820d`_)

* add ci (`3539ffb`_)

* add whitepsace (`3274b74`_)

* hacking with Benjamin (`23210a3`_)

* init project packaging (`5685aac`_)

* improve knot allocation (`8e4ad33`_)

* optimise starting weights (`1942d60`_)

* generate data for testing (`0d619ce`_)

* start hacking (`cd4026f`_)

.. _964e40e: https://github.com/nz-gravity/LogPSplinePSD/commit/964e40e8191ad20bdf3028bb268196312983058d
.. _7a0925c: https://github.com/nz-gravity/LogPSplinePSD/commit/7a0925cf8158fe5122ce68b9a41b9534af638099
.. _a43ee40: https://github.com/nz-gravity/LogPSplinePSD/commit/a43ee406b85b00fe480c36f9fbe1b45ce70a0683
.. _235c3ec: https://github.com/nz-gravity/LogPSplinePSD/commit/235c3ec5191c5c71952a820697d4416fc9b319e5
.. _9df1e5d: https://github.com/nz-gravity/LogPSplinePSD/commit/9df1e5d7527d08602a4402cb038e88c8aa474128
.. _027591f: https://github.com/nz-gravity/LogPSplinePSD/commit/027591fd3b4ecd334d784f25395d7bd5353c9ab2
.. _88c8f9b: https://github.com/nz-gravity/LogPSplinePSD/commit/88c8f9bc873be650cbcac1a2a3440db803b0afe5
.. _60af98d: https://github.com/nz-gravity/LogPSplinePSD/commit/60af98d50e3370107a7373018d72041a7f67e11d
.. _3a08b99: https://github.com/nz-gravity/LogPSplinePSD/commit/3a08b992d695f4bd9c9c8130989ee3de51341fed
.. _1fb79e8: https://github.com/nz-gravity/LogPSplinePSD/commit/1fb79e8689f87f89a4363d264bb1e33fbaf9217c
.. _b0cd698: https://github.com/nz-gravity/LogPSplinePSD/commit/b0cd6985070d56f217c4f63c6bc4f8da66c565ec
.. _fd9a95b: https://github.com/nz-gravity/LogPSplinePSD/commit/fd9a95bc154a1b7d009b3c4cb680a3cee9abfa5d
.. _c41038c: https://github.com/nz-gravity/LogPSplinePSD/commit/c41038cdc5ae858db11022f599862bf3becf4a69
.. _328e854: https://github.com/nz-gravity/LogPSplinePSD/commit/328e854df63dec4eacc4ec2738021c6c183489fb
.. _5877ec0: https://github.com/nz-gravity/LogPSplinePSD/commit/5877ec0c672fe51ad7013ebcdc931e30df990356
.. _20d3f39: https://github.com/nz-gravity/LogPSplinePSD/commit/20d3f393a5446bb1cd32f1661edd7993fff8ba97
.. _dc6469c: https://github.com/nz-gravity/LogPSplinePSD/commit/dc6469cff708fb172d5e90f2871ee57fb8e6c43a
.. _7f32bbb: https://github.com/nz-gravity/LogPSplinePSD/commit/7f32bbba2ddd96a0db3667ad1312b8acf7855a3d
.. _a061028: https://github.com/nz-gravity/LogPSplinePSD/commit/a06102836f95960b1699a073adbf441ea195b75c
.. _6b3905f: https://github.com/nz-gravity/LogPSplinePSD/commit/6b3905f03298d737dc1b940f7b4756dcbe122998
.. _cf42e6f: https://github.com/nz-gravity/LogPSplinePSD/commit/cf42e6f83eece3202eb747f09b1af55887082abb
.. _18d0075: https://github.com/nz-gravity/LogPSplinePSD/commit/18d007562a3e31dbed39a8c3b199252f951d03f7
.. _a2035bb: https://github.com/nz-gravity/LogPSplinePSD/commit/a2035bb40da74aa11dfd740af7b98af0a9d33ba5
.. _8fff25b: https://github.com/nz-gravity/LogPSplinePSD/commit/8fff25b4ae70f2627ca45c37ed57af842dd13353
.. _939cbdb: https://github.com/nz-gravity/LogPSplinePSD/commit/939cbdb650fbfdf460666ebb6f7e465f799e6e6e
.. _73fd427: https://github.com/nz-gravity/LogPSplinePSD/commit/73fd4276b6f44d68cfbb7fb16797be891f7e114a
.. _194fae8: https://github.com/nz-gravity/LogPSplinePSD/commit/194fae8d527bd7998dda38adaf0b96002c070414
.. _7231c3b: https://github.com/nz-gravity/LogPSplinePSD/commit/7231c3b1de002ee47b10286c4f799ae3551d4c40
.. _6641a63: https://github.com/nz-gravity/LogPSplinePSD/commit/6641a63c97f0c5392207fd56977ee37cf9811ac6
.. _98ae77a: https://github.com/nz-gravity/LogPSplinePSD/commit/98ae77ad38feaca0d65566f26d42e3adafe9f772
.. _d7121d6: https://github.com/nz-gravity/LogPSplinePSD/commit/d7121d6c1cd87a65355b4a6f6260578b90223339
.. _f818caa: https://github.com/nz-gravity/LogPSplinePSD/commit/f818caaa86467d5f26fb116a2c29c7a360ff41cf
.. _0666415: https://github.com/nz-gravity/LogPSplinePSD/commit/0666415347785d67b2865efe521648a7a89ee000
.. _4197563: https://github.com/nz-gravity/LogPSplinePSD/commit/4197563ebdd5da00a781dc22556eeb925f1cceaf
.. _922f870: https://github.com/nz-gravity/LogPSplinePSD/commit/922f87003a657d1578a98c3d3d803055f7969fe2
.. _4944aa1: https://github.com/nz-gravity/LogPSplinePSD/commit/4944aa1501d382d8ee4f6e06780c672e588b843d
.. _d93f36b: https://github.com/nz-gravity/LogPSplinePSD/commit/d93f36bcba5a70f2d90b40c3934de265f72cb65f
.. _c9e3c79: https://github.com/nz-gravity/LogPSplinePSD/commit/c9e3c790dff28a51bf9dc97b56bd63ccbcddd43b
.. _fda820d: https://github.com/nz-gravity/LogPSplinePSD/commit/fda820dd58f3072c86876d2a2ae218869f656f6e
.. _3539ffb: https://github.com/nz-gravity/LogPSplinePSD/commit/3539ffb0b1d87445201633488da63743454e0c7f
.. _3274b74: https://github.com/nz-gravity/LogPSplinePSD/commit/3274b74c1f0c59ea43825bdca177f99f8c8fe097
.. _23210a3: https://github.com/nz-gravity/LogPSplinePSD/commit/23210a35eb751832563a69101817ba906b82edba
.. _5685aac: https://github.com/nz-gravity/LogPSplinePSD/commit/5685aac389781eaeeadda6a1c31f2820b61cbed1
.. _8e4ad33: https://github.com/nz-gravity/LogPSplinePSD/commit/8e4ad33d4e99f20a2a76d40dd8539838ed5462ea
.. _1942d60: https://github.com/nz-gravity/LogPSplinePSD/commit/1942d6079393eb78ddcc07a7a4265805bcfcb010
.. _0d619ce: https://github.com/nz-gravity/LogPSplinePSD/commit/0d619ceba76869e3ec3b2d015987a77a1671cf19
.. _cd4026f: https://github.com/nz-gravity/LogPSplinePSD/commit/cd4026f9c50b1384a4cfba70cf8e67f938a254ac
