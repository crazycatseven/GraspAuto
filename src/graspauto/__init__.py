"""GraspAuto — real-time single-tap grasp authoring for VR/AR.

Reference implementation for the SIGGRAPH 2026 Poster submission:

    Point-M2AE object backbone  (point_m2ae_encoder.py)
    contact-graph utilities     (stage3_contact_graph.py)
    7-D grip-sphere token       (geom_palm_features.py)
    ResAE MANO latent           (mano_autoencoder.py)
    Perceiver-IO velocity net   (velocity_network.py)
    conditional flow matching   (flow_matching.py)
    MANO forward decoder        (mano_decoder.py)

See `paper/main.pdf` for the method overview and `DATA.md` for dataset setup.
"""
