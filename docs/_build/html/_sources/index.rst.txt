.. photomacros documentation master file, created by
   sphinx-quickstart on Tue Jan 21 20:07:17 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Photomacros Documentation
=====================================

Photomacros predicts images using Machine Learning.

Key Features
------------
.. hlist::
   :columns: 2

   * Raw Dataset Handling
   * Processed Dataset Management
   * Model Training
   * Predictions
   * Evaluation

Sections Overview
-----------------
.. grid:: 2

    .. grid-item-card::

        Photomacros
        ^^^^^^^^^^^

        Contains the data and configuration necessary to run the model.

        +++

        .. button-ref:: photomacros
            :expand:
            :color: primary
            :click-parent:

            Go to Photomacros

    .. grid-item-card::
        :margin: auto
        

        Modeling
        ^^^^^^^^

        Contains all the code for tasks such as training, prediction, and evaluation.

        +++

        .. button-ref:: modeling
            :expand:
            :color: primary
            :click-parent:

            Go to Modeling


.. toctree::
    :maxdepth: 2
    :titlesonly:
    :caption: Contents

    photomacros/index
    modeling/index