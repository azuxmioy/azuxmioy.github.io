function show() {
    document.getElementById('large_1').style.maxHeight = "200px";
    var images = document.querySelectorAll("#large_1 img");
    for(var i = 0; i < images.length; i++)
    {
      images[i].src = images[i].getAttribute('data-src');
    }
  }

function show() {
    document.getElementById('large_2').style.maxHeight = "200px";
    var images = document.querySelectorAll("#large_2 img");
    for(var i = 0; i < images.length; i++)
    {
      images[i].src = images[i].getAttribute('data-src');
    }
  } 