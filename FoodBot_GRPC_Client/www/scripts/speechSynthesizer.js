const synthesis = window.speechSynthesis;

function speak(message){
  const utter = new SpeechSynthesisUtterance(message);
  // the list of all available voices
  const voices = synthesis.getVoices();
  
  for(i = 0; i < voices.length; ++i) {
    if(voices[i].name === "Google English（US）") {
      //Google 國語（臺灣） (zh-TW)
      utter.voice = voices[i];
    }
  }
  
  utter.rate = 1;
  utter.pitch = 1;
  synthesis.speak(utter);
}