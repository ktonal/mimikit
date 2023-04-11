def demo():
    import mimikit as mmk
    from ipywidgets import widgets as W
    import IPython.display as ipd

    ipd.display(mmk.MMK_STYLE_SHEET)
    ipd.display(W.HTML(
        """
        <style>
        .container {
            width: 95% !important;
        }
        """
    ))

    app = mmk.ClusterizerApp()

    ipd.display(
        app.dataset_widget,
        app.clustering_widget,
        app.labels_widget
    )