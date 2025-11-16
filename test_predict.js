const fetch = require("node-fetch");

(async () => {
  const seq = [];
  for(let i=0;i<32;i++){ seq.push([10+i,20+i,30+i,40+i]); }
  const res = await fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({ sensors_sequence: seq })
  });
  console.log(await res.json());
})();
