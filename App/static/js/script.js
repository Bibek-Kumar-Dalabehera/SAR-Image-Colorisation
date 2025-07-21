WebFont.load({
    google: {
        families: ["Open Sans:300,300italic,400,400italic,600,600italic,700,700italic,800,800italic", "Space Mono:regular,italic,700,700italic", "Karla:regular,italic,700,700italic", "Frank Ruhl Libre:300,regular,500,700,900", "Playfair Display:regular,italic,700,700italic,900,900italic", "Lora:regular,italic,700,700italic"]
    }
});

!function(o, c) {
    var n = c.documentElement
      , t = " w-mod-";
    n.className += t + "js",
    ("ontouchstart"in o || o.DocumentTouch && c instanceof DocumentTouch) && (n.className += t + "touch")
}(window, document);

window.dataLayer = window.dataLayer || [];
function gtag() {
    dataLayer.push(arguments);
}
gtag('js', new Date());
gtag('config', 'UA-38766451-2', {
    'anonymize_ip': false
});

window.dataLayer = window.dataLayer || [];
function gtag() {
    dataLayer.push(arguments);
}
gtag('js', new Date());
gtag('set', 'developer_id.dZGVlNj', true);
gtag('config', 'G-CY6PEKQR3M');