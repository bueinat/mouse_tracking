import React, { useState } from 'react';
import Tabs from "./components/Tabs";
import "./App.css";

// import SplitPane, {
//   Divider,
//   SplitPaneBottom,
//   SplitPaneLeft,
//   SplitPaneRight,
//   SplitPaneTop,
// } from "./SplitPane";

 

var namesList = []

function App() {
  const [name, setName] = useState('')
  const onSubmit = (e) => {
    e.preventDefault();
    console.log("submitted!")
    namesList = [...namesList, name]
    console.log(namesList)
    setName('')
  }

  // return (
  // <SplitPane split="vertical">
  //   <Pane initialSize="200px">You can use a Pane component</Pane>
  //   <div>or you can use a plain old div</div>
  //   <Pane initialSize="25%" minSize="10%" maxSize="500px">
  //     Using a Pane allows you to specify any constraints directly
  //   </Pane>
  // </SplitPane>
  // )
  return (
      <Tabs>
        <div label="Video">
          <form onSubmit={onSubmit}>
            <label>
              Name: 
              <input type="text" value={name} onChange={e => {
                // e.defaultPrevented()
                setName(e.target.value)
                // e.target.value = ''
              }}
                placeholder='type your name' />
              <input type="submit" value="Submit" />
            </label>
          </form>
          See ya later, <em>Alligator</em>!
        </div>
        <div label="Analyze">
        <ul>{namesList.map((n) =>
            <li>{n}</li>)}</ul>
          After 'while, <em>Crocodile</em>!
        </div>
        <div label="Group and Export">
          Nothing to see here, this tab is <em>extinct</em>!
        </div>
      </Tabs>
    // </div>
  );
}

export default App;
