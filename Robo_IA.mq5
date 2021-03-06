//+------------------------------------------------------------------+
//|                                                           Client |
//|                                      Rafael Fenerick (adaptação) |
//|                       programming & development - Alexey Sergeev |
//+------------------------------------------------------------------+
#property copyright "© 2006-2016 Alexey Sergeev"
#property link      "profy.mql@gmail.com"
#property version   "1.00"

#include <socketlib.mqh>
#include <funcoes.mqh>
#include <Trade\Trade.mqh>
CTrade trade; 

input string Host="127.0.0.1";   // Host
input ushort Port=8888;          // Porta
input int    Time=3;             // Tempo entre mensagens (segundos)
string Msg="Hello World!"; // Mensagem
bool comprado = false;
bool vendido = false;
int magic_number1 = 123456;   // Nº mágico do robô
SOCKET64 client=INVALID_SOCKET64; // client socket
ref_sockaddr srvaddr={0};

//------------------------------------------------------------------	OnInit
int OnInit()
  {
// fill the structure for the server address
   string dados = indicadores();
   Msg = dados;
   trade.SetTypeFilling(ORDER_FILLING_RETURN);
   trade.SetDeviationInPoints(50);
   trade.SetExpertMagicNumber(magic_number1);
   inicializacao();
   char ch[]; StringToCharArray(Host,ch);
   sockaddr_in addrin;
   addrin.sin_family=AF_INET;
   addrin.sin_addr.u.S_addr=inet_addr(ch);
   addrin.sin_port=htons(Port);
   srvaddr.in=addrin;

   

   return INIT_SUCCEEDED;
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick(void)
  {
//---
      
      if(TemosNovaVela())
       {
         string dados = indicadores();
         Msg = dados;
         Print("Envio: ",velas[1].time);
         OnTimerh();
       }

//---
  }

//------------------------------------------------------------------	OnDeinit
void OnDeinit(const int reason) { CloseClean(); EventKillTimer();}
//------------------------------------------------------------------	OnTick
void receber()
{
      int res=0;
      char wsaData[]; ArrayResize(wsaData,sizeof(WSAData));
      res=WSAStartup(MAKEWORD(2,2),wsaData);
      if(res!=0) { Print("-WSAStartup failed error: "+string(res)); return; }

      // create socket
      client=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
      if(client==INVALID_SOCKET64) { Print("-Create failed error: "+WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

      // set to nonblocking mode
      int non_block=1;
      res=ioctlsocket(client,(int)FIONBIO,non_block);
      if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); CloseClean(); return; }

      Print("create socket OK");
                     //Print("send "+Symbol()+" msg to server");
      uchar data_received[];
      if(Receive(data_received)>0) // receive data
        {
         string msg=CharArrayToString(data_received);
         string result[];
         string sep = "-";
         ushort u_sep; 
         u_sep=StringGetCharacter(sep,0);
         StringSplit(msg,u_sep,result);
         Print(msg);
         string hora =result[1];
         Print("Hora: ",result[1]," Tipo: ",result[0]);
         if(result[0] == "compra" && PositionSelect(_Symbol) == false)
           {
            comprado = true;
            double preco = tick.bid;
            double SL = 300;
            double TK = 300;
            double PRC = NormalizeDouble(preco*Point(),_Digits);             // Preco de compra
            double STL = NormalizeDouble((preco - SL)*Point(),_Digits); // Stop Loss
            double TKP = NormalizeDouble((preco + TK)*Point(),_Digits); // Alvo de Ganho - Take Profit
            
            if(trade.Buy(1, _Symbol,PRC,STL,TKP,hora))
            {
              Print("Ordem de compra sem falha. ResultCode: ", trade.ResultRetcode(), " RetcodeDescription: ", trade.ResultRetcodeDescription());
            }
            else
            {
              Print("Ordem de compra  COM falha. ResultCode: ", trade.ResultRetcode(), " RetcodeDescription: ", trade.ResultRetcodeDescription());
            }
           }
         if(result[0] == "venda" && PositionSelect(_Symbol) == false)
           {
            vendido = true;
            double preco = tick.ask;
            double SL = 300;
            double TK = 300;
            double PRC = NormalizeDouble(preco*Point(),_Digits);             // Preco de compra
            double STL = NormalizeDouble((preco + SL)*Point(),_Digits); // Stop Loss
            double TKP = NormalizeDouble((preco - TK)*Point(),_Digits); // Alvo de Ganho - Take Profit
            trade.Sell(1,_Symbol,preco,STL,TKP,hora);
            
           }
         if(result[0] == "compra" && vendido)
           {
             vendido = false;
             trade.PositionClose(_Symbol,10);
           }
         if(result[0] == "venda" && comprado)
           {
             comprado = false;
             trade.PositionClose(_Symbol,10);
           }
        }

}
void OnTimerh()
  {
   if(client!=INVALID_SOCKET64) // if the socket is already created, send
     {
      uchar data[];
      StringToCharArray(Msg, data);
      if(sendto(client,data,ArraySize(data),0,srvaddr.ref,ArraySize(srvaddr.ref))==SOCKET_ERROR)
        {
         int err=WSAGetLastError();
         if(err!=WSAEWOULDBLOCK) { Print("-Send failed error: "+WSAErrorDescript(err)); CloseClean(); }
        }
      else
       {
                     //Print("send "+Symbol()+" msg to server");
                    uchar data_received[];
                     if(Receive(data_received)>0) // receive data
                       {
                        string msg=CharArrayToString(data_received);
                        string result[];
                        string sep = "-";
                        ushort u_sep; 
                        u_sep=StringGetCharacter(sep,0);
                        StringSplit(msg,u_sep,result);
                        Print(msg);
                        string hora =result[1];
                        Print("Hora: ",result[1]," Tipo: ",result[0]);
                        if(result[0] == "compra" && PositionSelect(_Symbol) == false)
                          {
                           comprado = true;
                           double preco = tick.bid;
                           double SL = 300;
                           double TK = 300;
                           double PRC = NormalizeDouble(preco*Point(),_Digits);             // Preco de compra
                           double STL = NormalizeDouble((preco - SL)*Point(),_Digits); // Stop Loss
                           double TKP = NormalizeDouble((preco + TK)*Point(),_Digits); // Alvo de Ganho - Take Profit
                           
                           if(trade.Buy(1, _Symbol,PRC,STL,TKP,hora))
                           {
                             Print("Ordem de compra sem falha. ResultCode: ", trade.ResultRetcode(), " RetcodeDescription: ", trade.ResultRetcodeDescription());
                           }
                           else
                           {
                             Print("Ordem de compra  COM falha. ResultCode: ", trade.ResultRetcode(), " RetcodeDescription: ", trade.ResultRetcodeDescription());
                           }
                          }
                        if(result[0] == "venda" && PositionSelect(_Symbol) == false)
                          {
                           vendido = true;
                           double preco = tick.ask;
                           double SL = 300;
                           double TK = 300;
                           double PRC = NormalizeDouble(preco*Point(),_Digits);             // Preco de compra
                           double STL = NormalizeDouble((preco + SL)*Point(),_Digits); // Stop Loss
                           double TKP = NormalizeDouble((preco - TK)*Point(),_Digits); // Alvo de Ganho - Take Profit
                           trade.Sell(1,_Symbol,preco,STL,TKP,hora);
                           
                          }
                        if(result[0] == "compra" && vendido)
                          {
                            vendido = false;
                            trade.PositionClose(_Symbol,10);
                          }
                        if(result[0] == "venda" && comprado)
                          {
                            comprado = false;
                            trade.PositionClose(_Symbol,10);
                          }
                       }
        }
     }
   else // create a client socket
     {
      int res=0;
      char wsaData[]; ArrayResize(wsaData,sizeof(WSAData));
      res=WSAStartup(MAKEWORD(2,2),wsaData);
      if(res!=0) { Print("-WSAStartup failed error: "+string(res)); return; }

      // create socket
      client=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);
      if(client==INVALID_SOCKET64) { Print("-Create failed error: "+WSAErrorDescript(WSAGetLastError())); CloseClean(); return; }

      // set to nonblocking mode
      int non_block=1;
      res=ioctlsocket(client,(int)FIONBIO,non_block);
      if(res!=NO_ERROR) { Print("ioctlsocket failed error: "+string(res)); CloseClean(); return; }

      Print("create socket OK");
     }
  }
//------------------------------------------------------------------	Receive
int Receive(uchar &rdata[]) // Receive until the peer closes the connection
  {
   if(client==INVALID_SOCKET64) return 0; // if the socket is still not open

   char rbuf[512]; int rlen=512; int r=0,res=0;
   do
     {
      res=recv(client,rbuf,rlen,0);
      if(res<0)
        {
         int err=WSAGetLastError();
         if(err!=WSAEWOULDBLOCK) { Print("-Receive failed error: "+string(err)+" "+WSAErrorDescript(err)); CloseClean(); return -1; }
         break;
        }
      if(res==0 && r==0) { Print("-Receive. connection closed"); CloseClean(); return -1; }
      r+=res; ArrayCopy(rdata,rbuf,ArraySize(rdata),0,res);
     }
   while(res>0 && res>=rlen);
   Print("Teste: ",res);
   return r;
  }
//------------------------------------------------------------------	CloseClean
void CloseClean() // close socket
  {
   if(client!=INVALID_SOCKET64)
     {
      if(shutdown(client,SD_BOTH)==SOCKET_ERROR) Print("-Shutdown failed error: "+WSAErrorDescript(WSAGetLastError()));
      closesocket(client); client=INVALID_SOCKET64;
     }

   WSACleanup();
   Print("close socket");
  }
//+------------------------------------------------------------------+
bool TemosNovaVela()
  {
//--- memoriza o tempo de abertura da ultima barra (vela) numa variável
   static datetime last_time=0;
//--- tempo atual
   datetime lastbar_time= (datetime) SeriesInfoInteger(Symbol(),Period(),SERIES_LASTBAR_DATE);

//--- se for a primeira chamada da função:
   if(last_time==0)
     {
      //--- atribuir valor temporal e sair
      last_time=lastbar_time;
      return(false);
     }

//--- se o tempo estiver diferente:
   if(last_time!=lastbar_time)
     {
      //--- memorizar esse tempo e retornar true
      last_time=lastbar_time;
      return(true);
     }
//--- se passarmos desta linha, então a barra não é nova; retornar false
   return(false);
  }
//------------------------------------------------------------- 
/* Inicio Compra a Mercado */
void CompraAMercado(double preco,double SL, double TK,double num_lots) // bser na documentação ordem das variaveis!!!
  {
      double PRC = NormalizeDouble(preco*Point(),_Digits);             // Preco de compra
      double STL = NormalizeDouble((preco - SL)*Point(),_Digits); // Stop Loss
      double TKP = NormalizeDouble((preco + TK)*Point(),_Digits); // Alvo de Ganho - Take Profit
      Print(" Compra: Take Profit: ", TK, " Stop Loss: ", SL , " Compra: ", PRC);
      //trade.BuyLimit(num_lots,PRC,_Symbol,STL,TKP);
      if(trade.Buy(num_lots, _Symbol,PRC,STL,TKP,""))
       {
           Print("Ordem de compra sem falha. ResultCode: ", trade.ResultRetcode(), " RetcodeDescription: ", trade.ResultRetcodeDescription());
       }
       else
       {
           Print("Ordem de compra  COM falha. ResultCode: ", trade.ResultRetcode(), " RetcodeDescription: ", trade.ResultRetcodeDescription());
       }
  }
  
  /* Inicio Venda a Mercado */
void VendaAMercado(double preco,double SL, double TK,double num_lots)
  {
      //preco = velas[1].low;
      double PRC = NormalizeDouble(preco*Point(),_Digits);            // Preço para Venda
      double STL = NormalizeDouble((preco + SL)*Point(),_Digits);       // Preço Stop Loss
      double TKP = NormalizeDouble((preco - TK)*Point(),_Digits);       // Alvo de Ganho - Take Profit
      Print(" Venda: Take Profit: ", TK, " Stop Loss: ", SL , " Venda: ", PRC);
      if(trade.Sell(num_lots, _Symbol,PRC,STL,TKP,""))
       {
           Print("Ordem de compra sem falha. ResultCode: ", trade.ResultRetcode(), " RetcodeDescription: ", trade.ResultRetcodeDescription());
       }
      else
       {
           Print("Ordem de compra  COM falha. ResultCode: ", trade.ResultRetcode(), " RetcodeDescription: ", trade.ResultRetcodeDescription());
       } 
 }