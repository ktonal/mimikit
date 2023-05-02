def demo():
    """# Generate From Checkpoint

    ### How to interact with the waveforms:

    - Navigation:
        * `Ctrl + wheel`: zoom
        * `SHIFT + dbl-click`: reset zoom
        * `SHIFT + wheel`: scroll waveform
        * `arrow left/right`: move playhead left/right
        * `SHIFT + arrow left/right`: move playhead left/right a lot.
    - Controls:
        * `dbl-click`: play from there
        * `SPACE BAR`: play/pause
    - Prompts:
        * `Ctrl + click`: add prompt
        * `SHIFT + click` on a prompt: remove prompt
        * `drag` a prompt's handle to move it left/right
    """
    import mimikit as mmk

    mmk.GenerateFromCheckpointView(root_dir="./")

    """----------------------------"""
