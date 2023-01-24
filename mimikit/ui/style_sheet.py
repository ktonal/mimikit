import ipywidgets as W

__all__ = [
    "MMK_STYLE_SHEET"
]


MMK_STYLE_SHEET = W.HTML(
"""
<style>

.picker-button {
    border: 0px !important;
    border-radius: 5px !important;
    background-color: white !important;
    text-align: left !important;
}
.not-a-button:hover {
    box-shadow: none !important;
}
.selected-button {
    background-color: lightgreen !important;
}
.gray-label {
    color: gray !important;
}

.selected {
    overflow-wrap: anywhere !important;
    text-color: black !important;
    padding: 2px !important;
    opacity: 1 !important;
}

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

.jupyter-widgets.widget-tab > .p-TabBar .p-TabBar-tab {
    flex: 0 1 auto
}
</style>
"""
)
