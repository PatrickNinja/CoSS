# CoSS: leveraging statement semantics for code summarization
Check the paper [CoSS: leveraging statement semantics for code summarization](https://drive.google.com/file/d/1Hw8as_GjRFkUh6Gf_x_71kUDo2pFVcO2/view?usp=sharing).
### Dependences
- python 3.7
- torch == 1.4.0
- transformers == 3.5.0


### Data
 The Java dataset is collected from [CodeSearchNet](https://arxiv.org/pdf/1909.09436.pdf).


To download and pre-process java dataset:

```
mkdir dataset
cd dataset
wget -q https://s3.amazonaws.com/code-search-net/CodeSearchNet/v2/java.zip
unzip -qq java.zip

cd ..
python data_process_java.py
```

Python and Solidity dataset can be found here: [dataset](https://drive.google.com/drive/folders/1of6Q9sYaUVCjn4xJSrjcyclXTnjBmCYl?usp=sharing). Put the dataset folder under the root directory and preprocess:
```
python data_process_python.py
python data_process_solidity.py
```
### Model Training
An example of model training settings:
```
lang=java #programming languages, could be java, solidity, or python
data_dir=./processed_data

python train.py \
    --do_train \
    --do_eval \
    --do_lower_case \
    --train_filename $data_dir/$lang/train.jsonl \
    --dev_filename $data_dir/$lang/valid.jsonl \
    --output_dir model/$lang \
    --max_source_length 256 \
    --max_target_length 48 \
    --beam_size 10 \
    --train_batch_size 5 \
    --eval_batch_size 5 \
    --learning_rate 5e-5 \
    --num_train_epochs 10
```

### Load Model and Generate Example Outputs
When we get the trained model, we can generate example outputs:

```
python output_$lang.py 
```
Example outputs:
```
Code: public void removeimageview(cubeimageview imageview) { if (null == imageview || null == mfirstimageviewholder) { return; } imageviewholder holder = mfirstimageviewholder; do { if (holder.contains(imageview)) { // make sure entry is right. if (holder == mfirstimageviewholder) { mfirstimageviewholder = holder.mnext; } if (null != holder.mnext) { holder.mnext.mprev = holder.mprev; } if (null != holder.mprev) { holder.mprev.mnext = holder.mnext; } } } while ((holder = holder.mnext) != null); }
Original Comment: remove the imageview from imagetask
Generated Comment: remove the current view.
========================================
Code: protected string getquery() { final stringbuilder ret = new stringbuilder(); try { final string clazzname; if (efapssystemconfiguration.get().containsattributevalue("org.efaps.kernel.index.querybuilder")) { clazzname = efapssystemconfiguration.get().getattributevalue("org.efaps.kernel.index.querybuilder"); } else { clazzname = "org.efaps.esjp.admin.index.lucencequerybuilder"; } final class<?> clazz = class.forname(clazzname, false, efapsclassloader.getinstance()); final object obj = clazz.newinstance(); final method method = clazz.getmethod("getquery4dimvalues", string.class, list.class, list.class); final object newquery = method.invoke(obj, getcurrentquery(), getincluded(), getexcluded()); ret.append(newquery); } catch (final efapsexception | classnotfoundexception | instantiationexception | illegalaccessexception | nosuchmethodexception | securityexception | illegalargumentexception | invocationtargetexception e) { indexsearch.log.error("catched", e); ret.append(getcurrentquery()); } return ret.tostring(); }
Original Comment: gets the query.
Generated Comment: get the query instance.
========================================
Code: private void handlehttpclienterrorsforbackend(final httprequest clientrequest, final exception e) { /* notify error handler that we got an error. */ errorhandler.accept(e); /* increment our error count. */ errorcount.incrementandget(); /* create the error message. */ final string errormessage = string.format("unable to make request %s ", clientrequest.address()); /* log the error. */ logger.error(errormessage, e); /* don't send the error to the client if we already handled this, i.e., timedout already. */ if (!clientrequest.ishandled()) { clientrequest.handled(); /* notify the client that there was an error. */ clientrequest.getreceiver().error(string.format("\"%s\"", errormessage)); } }
Original Comment: handle errors.
Generated Comment: handles an error from the server.
```
