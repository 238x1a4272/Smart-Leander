import React,{useState} from "react";
import {SafeAreaView,View,Text,TextInput,TouchableOpacity,ScrollView,StyleSheet,Alert,ActivityIndicator} from "react-native";

const API_BASE_URL="http://YOUR-SERVER-IP:5000";

const fields=[
 ["Gender","0 or 1"],["Married","0 or 1"],["Dependents","0, 1, 2 or 3"],["Education","0 or 1"],
 ["Self_Employed","0 or 1"],["ApplicantIncome","e.g. 5000"],["CoapplicantIncome","e.g. 1500"],
 ["LoanAmount","e.g. 120"],["Loan_Amount_Term","e.g. 360"],["Credit_History","0 or 1"],["Property_Area","0, 1 or 2"]
];

export default function App(){
 const [values,setValues]=useState(Object.fromEntries(fields.map(([k])=>[k,""])));
 const [loading,setLoading]=useState(false);
 const [result,setResult]=useState(null);

 const update=(k,v)=>setValues({...values,[k]:v});
 const submit=async()=>{
   if(Object.values(values).some(v=>v.trim()==="")) return Alert.alert("Missing information","Please complete all fields.");
   setLoading(true); setResult(null);
   try{
     const body=new URLSearchParams(values).toString();
     const res=await fetch(API_BASE_URL+"/submit",{method:"POST",headers:{"Content-Type":"application/x-www-form-urlencoded"},body});
     const text=await res.text();
     const approved=text.includes("Loan Approved");
     const rejected=text.includes("Loan Not Approved");
     if(!approved && !rejected) throw new Error("Unexpected server response");
     setResult(approved?"approved":"rejected");
   }catch(e){Alert.alert("Connection error","Check API_BASE_URL and make sure the Flask server is running.");}
   finally{setLoading(false);}
 };

 return <SafeAreaView style={s.safe}>
   <ScrollView contentContainerStyle={s.container}>
     <View style={s.hero}>
       <Text style={s.logo}>SMART LEANDER</Text>
       <Text style={s.title}>Loan eligibility</Text>
       <Text style={s.subtitle}>Enter the applicant details and get the prediction from your trained machine-learning model.</Text>
     </View>
     <View style={s.card}>
       {fields.map(([k,hint])=><View key={k} style={s.field}>
         <Text style={s.label}>{k.replaceAll("_"," ")}</Text>
         <TextInput value={values[k]} onChangeText={v=>update(k,v)} placeholder={hint} keyboardType="numeric" style={s.input}/>
       </View>)}
       <TouchableOpacity style={s.button} onPress={submit} disabled={loading}>
         {loading?<ActivityIndicator color="#fff"/>:<Text style={s.buttonText}>CHECK ELIGIBILITY</Text>}
       </TouchableOpacity>
     </View>
     {result && <View style={[s.result,result==="approved"?s.good:s.bad]}>
       <Text style={s.resultIcon}>{result==="approved"?"✓":"!"}</Text>
       <Text style={s.resultTitle}>{result==="approved"?"Loan Approved":"Loan Not Approved"}</Text>
       <Text style={s.resultText}>Prediction returned by the project's decision-tree model.</Text>
     </View>}
     <Text style={s.note}>This app provides a machine-learning prediction and should not be treated as a lending decision.</Text>
   </ScrollView>
 </SafeAreaView>
}
const s=StyleSheet.create({
 safe:{flex:1,backgroundColor:"#F4F7F5"},container:{padding:18,paddingBottom:40},
 hero:{backgroundColor:"#173B32",borderRadius:26,padding:24,marginBottom:16},logo:{color:"#BCE7D1",fontWeight:"900",letterSpacing:2,fontSize:13},
 title:{color:"#fff",fontSize:32,fontWeight:"900",marginTop:10},subtitle:{color:"#DCE9E3",fontSize:15,lineHeight:22,marginTop:8},
 card:{backgroundColor:"#fff",borderRadius:22,padding:18},field:{marginBottom:13},label:{fontSize:14,fontWeight:"800",textTransform:"capitalize",marginBottom:6,color:"#25332D"},
 input:{borderWidth:1,borderColor:"#D7DED9",borderRadius:12,padding:13,fontSize:16,backgroundColor:"#FAFCFB"},
 button:{backgroundColor:"#173B32",borderRadius:14,padding:16,alignItems:"center",marginTop:8},buttonText:{color:"#fff",fontWeight:"900",letterSpacing:.5},
 result:{borderRadius:20,padding:22,marginTop:16},good:{backgroundColor:"#DDF4E6"},bad:{backgroundColor:"#F8DFDF"},
 resultIcon:{fontSize:32,fontWeight:"900"},resultTitle:{fontSize:24,fontWeight:"900",marginTop:4},resultText:{marginTop:5,color:"#46534D"},
 note:{fontSize:12,color:"#68736E",lineHeight:18,margin:18,textAlign:"center"}
});