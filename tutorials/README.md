Welcome! This is the home page for the Winter 2016 edition of the "Neural Data Analysis" course.

=== Contents ===

== Reference ==

   * [Principles of (neural) data analysis](analysis:nsb2015:week0)

== Fundamentals ==

   * [Module 1: Setting up (MATLAB, paths, GitHub, accessing data; Week 1)](analysis:course-w16:week1)
   * [Module 2: Introduction to neural data formats and preprocessing (Week 2)](analysis:course-w16:week2)
   * [Module 3: Visualizing raw neural data in MATLAB (Week 3)](analysis:course-w16:week3long)
 
== Time series data data basics ==

   * [Module 4: Anatomy of time series data, sampling theory (Week 4)](analysis:course-w16:week4)
   * [Module 5: Fourier series, transforms, power spectra (Week 4)](analysis:course-w16:week5)
   * [Module 6: Filtering: filter design, use, caveats (Week 5)](analysis:course-w16:week6)
   * [Module 7: Time-frequency analysis: spectrograms (Week 5)](analysis:course-w16:week7)

== Spike data basics ==

   * [Module 8: Spike sorting](analysis:course-w16:week8) 
   (we will probably skip this one, but you are welcome to go through it yourself.)
   * [Module 9: Spike train analysis: firing rate, interspike interval distributions, auto- and crosscorrelations (Week 6)](analysis:course-w16:week9)
   * [Module 10: Spike train analysis II: tuning curves, encoding, decoding (Week 7)](analysis:course-w16:week10)


== Intermediate topics ==

   * [Module 11: Interactions between multiple signals: coherence, Granger causality, and phase-slope index (Week 8)](analysis:course-w16:week11)
   * [Module 12: Time-frequency analysis II: cross-frequency coupling (Week 9)](analysis:course-w16:week12)
   * [Module 13: Spike-field relationships: spike-triggered average, phase locking, phase precession (Week 10)](analysis:course-w16:week13)
   * [Module 14: Classification of ensemble spiking patterns](analysis:course-w16:week14) (likely skip)

== Advanced topics ==

  * [Module 15: Two-step Bayesian decoding with dynamic spatial priors](analysis:course-w16:week15) (likely skip)
  * [Module 16: Pairwise co-occurrence](analysis:course-w16:week16) (likely skip)

== Other topics ==

  * Git: conflict resolution, undo's, writing good commit messages, issue tracking, branching (on request)
  * Top-level analysis workflows for handling multiple subjects and sessions (on request)
  * Exporting MATLAB data to R (on request)
  * MATLAB tools: GUI design tool, debugger, profiler (on request)

=== Prerequisites ===

Basic familiarity with MATLAB. Depending on your background and programming experience you might find the following resources helpful:

  * Textbook: Wallisch, MATLAB for Neuroscientists
  * ["Getting Started with MATLAB" Primer](http://www.mathworks.com/help/matlab/getting-started-with-matlab.html?s_cid=learn_doc). 
  * [Cody](http://www.mathworks.com/matlabcentral/about/cody/),
   a continually expanding set of problems with solutions to work through, with a points system to track your progress

If you are unsure, take a look at the table of contents of the Primer. 
If there are things you don't recognize, use the Primer itself, 
or Chapter 2 of the MATLAB for Neuroscientists book to get up to speed.

Regardless of your MATLAB abilities, some great ways to keep learning are:

  * [Mathworks staff blogs](http://blogs.mathworks.com/), 
  especially "Loren on the Art of MATLAB" is a treasure trove of tips and tricks
  * [MATLAB questions on StackOverflow](http://stackoverflow.com/questions/tagged/matlab), 
  a Q&A site where you can browse previous questions and add new ones

If you have no formal training in computer programming 
(i.e. you have never taken a "Intro to Computer Science" or "Introductory Programming" type course) 
you will almost certainly find what follows in this course less frustrating if you do the pen-and-paper exercises in this
 [short chapter](http://sites.tufts.edu/rodrego/files/2011/03/Secrets-of-Computer-Power-Revealed-2008.pdf)
  by Daniel Dennett ("The Secrets of Computer Power Revealed") before you embark on the MATLAB primer linked to above.

=== Resources ===

This course is "standalone", but the following textbooks provide more in-depth treatment of some of the topics.

  * Textbook: Leis, Digital Signal Processing using MATLAB for Students and Researchers
  * Textbook: Johnston and Wu, Foundations of Cellular Neurophysiology
  * Textbook: Dayan & Abbott, Theoretical Neuroscience

=== What this course is ===

Overall, the course is designed to provide hands-on experience with management, 
visualization, and analysis of neural data. 
Becoming skilled at these things is a rate-limiting step for many graduate projects requiring analysis. 
Even if your work only requires rudimentary analysis, awareness of what else can be done 
and how to do it well is valuable, for instance when evaluating the work of others in the literature!

To do so, the focus is on introducing some commonly used tools, 
such as GitHub and relevant functionality within MATLAB -- 
and then to actually use these on real data sets. 
Initially, those data sets will be local field potentials and spiking data recorded from various brain areas 
in freely moving rodents in the van der Meer lab; 
however, a crucial goal of the course is that after some initial steps you will use your own data. 
(If you don't have your own, you can do everything with data provided here.)

We will make contact with a few concepts from computer science, signal processing, and statistics. 
However, the focus is on making initial steps that work and getting pointers to more complete treatment, 
rather than a thorough theoretical grounding.
 Nevertheless, to make sure that what you learn is not tied to specific data sets only, a number of 
 [principles](analysis:nsb2015:week0) of data analysis 
 -- applicable to any project of sufficient complexity 
 -- will be referenced throughout the course. 
 You are invited to think of these and others, not only as you progress through this course, 
 but especially as you organize your own data analyses and read analyses performed by others.

=== What this course is not ===

This course will provide a brief introduction to a number of concepts 
which are themselves the subject of multiple courses and voluminous textbooks. 
These include signal processing topics such as Fourier analysis and filter design, 
computer science concepts such as object-oriented programming and binary data formats, 
and a number of statistical ideas and tools. 
Be aware that if any of these are particularly important to your research, 
you should consider taking more in-depth coursework and/or working through relevant textbooks on your own: 
this short tutorial cannot replace such courses!

=== Evaluation ===

Most modules finish with a challenge (or several), i
n which you are invited to implement some of the ideas in the module yourself. 
Pick one such challenge from the first half of modules (1-7) and another from the second half (9-16). 
Submit your code for two chosen challenges to a GitHub repository you created, along with documentation: 
instructions on what it is supposed to do, how to make it run if applicable, and comments explaining how the code works.

=== Note for Linux users ===

The lab codebase is set up for machines running 64-bit Windows 7 or Mac %%OS%% X. 
If you want to use Linux or some other OS you will probably need to compile 
some of the low-level loading functions yourself.
 Some pointers for this are provided in subsequent modules when loading is introduced.

=== Acknowledgments ===

The architecture of the code used in this course was inspired by a similar set of code by my post-doctoral mentor, 
[A. David Redish](http://redishlab.neuroscience.umn.edu/); 
several of the data types and functions are re-implementations of Redish lab functions of the same name.
 Major contributions to the codebase were made by Alyssa Carey 
 (a MSc student and research assistant in the lab) and Youki Tanaka (current PhD student). 