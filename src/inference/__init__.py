"""Portable final-model training and inference interfaces."""

from .final_predictor import FinalModelBundle, load_bundle, save_bundle, train_final_bundle

__all__ = ["FinalModelBundle", "load_bundle", "save_bundle", "train_final_bundle"]
