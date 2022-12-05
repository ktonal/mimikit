import ipywidgets as W

__all__ = [
    "MMK_STYLE_SHEET"
]


MMK_STYLE_SHEET = W.HTML(
"""
<style>

/////////////  FILE PICKER

.not-a-button {
    border: 0px !important;
    border-radius: 5px !important;
    background-color: rgb(255, 255, 255) !important;
    text-align: left !important;
}
.not-a-button:hover {
    box-shadow: initial !important;
}
.selected-button {
    background-color: lightgreen !important;
}
.selected {
    overflow: scroll !important;
    
}

/////////////  TOOLTIPS

.tltp {
        border-radius: 8px 8px !important;
        background-color: rgb(247, 228, 0) !important;
        color: rgb(0, 0, 0) !important;
    }
.tltp i {
    font-size: 8px !important;
    position: absolute !important;
    top: 2px !important;
    left: 8px !important;
}
</style>
"""
)
